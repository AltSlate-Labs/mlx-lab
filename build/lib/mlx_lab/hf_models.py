"""Hugging Face model discovery and inspection client."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from ._version import __version__

JsonObject = dict[str, Any]
Opener = Callable[..., Any]

HF_HUB_BASE_URL = "https://huggingface.co"
_PARAM_PATTERN = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)([bm])(?![a-z])", re.IGNORECASE)


class APIRequestError(RuntimeError):
    """Raised when a Hugging Face API request fails."""


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if isinstance(value, str):
        return [value] if value else []
    return [str(value)]


def _normalize_token(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = _normalize_token(value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _extract_tag_values(tags: list[str], prefixes: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    normalized_prefixes = tuple(prefix.lower() for prefix in prefixes)
    for tag in tags:
        if ":" not in tag:
            continue
        key, value = tag.split(":", 1)
        if key.lower() in normalized_prefixes and value:
            values.append(value)
    return values


def _extract_license(card_data: JsonObject, tags: list[str]) -> str | None:
    license_values = _as_str_list(card_data.get("license"))
    license_values.extend(_extract_tag_values(tags, ("license",)))
    for value in license_values:
        if value:
            return value
    return None


def _extract_summary(item: JsonObject, card_data: JsonObject) -> str:
    for key in ("summary", "description", "model_summary", "pretty_name"):
        value = card_data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("description", "pipeline_tag"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_task_tag(item: JsonObject, tags: list[str]) -> str | None:
    pipeline_tag = item.get("pipeline_tag") or item.get("pipelineTag")
    if isinstance(pipeline_tag, str) and pipeline_tag:
        return pipeline_tag
    task_values = _extract_tag_values(tags, ("task", "pipeline_tag"))
    return task_values[0] if task_values else None


def _extract_architecture(item: JsonObject, tags: list[str]) -> str | None:
    config = item.get("config")
    if isinstance(config, dict):
        model_type = config.get("model_type") or config.get("modelType")
        if isinstance(model_type, str) and model_type:
            return model_type
        architectures = config.get("architectures")
        if isinstance(architectures, list) and architectures:
            first = architectures[0]
            if isinstance(first, str) and first:
                return first

    architecture_tags = _extract_tag_values(tags, ("architecture",))
    if architecture_tags:
        return architecture_tags[0]
    return None


def _extract_number(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _parse_param_token(raw_token: str, unit: str) -> int | None:
    try:
        base = float(raw_token)
    except ValueError:
        return None
    unit_lower = unit.lower()
    if unit_lower == "b":
        return int(base * 1_000_000_000)
    if unit_lower == "m":
        return int(base * 1_000_000)
    return None


def _extract_param_count_from_text(text: str) -> int | None:
    matches = _PARAM_PATTERN.findall(text)
    if not matches:
        return None
    counts: list[int] = []
    for raw_value, unit in matches:
        parsed = _parse_param_token(raw_value, unit)
        if parsed:
            counts.append(parsed)
    if not counts:
        return None
    return max(counts)


def _extract_parameter_count(item: JsonObject, tags: list[str]) -> int | None:
    for key in ("parameter_count", "parameters"):
        direct_value = _extract_number(item.get(key))
        if direct_value is not None:
            return direct_value

    card_data = item.get("cardData")
    if isinstance(card_data, dict):
        for key in ("parameter_count", "parameters"):
            direct_value = _extract_number(card_data.get(key))
            if direct_value is not None:
                return direct_value

    joined_tag_text = " ".join(tags)
    from_tags = _extract_param_count_from_text(joined_tag_text)
    if from_tags is not None:
        return from_tags

    model_id = str(item.get("id") or item.get("modelId") or "")
    return _extract_param_count_from_text(model_id)


def _size_class_for_params(parameter_count: int | None) -> str:
    if parameter_count is None:
        return "unknown"
    if parameter_count <= 3_000_000_000:
        return "small"
    if parameter_count <= 7_000_000_000:
        return "medium"
    if parameter_count <= 13_000_000_000:
        return "large"
    return "xlarge"


def _matches_filter(filter_value: str | None, candidate_values: list[str]) -> bool:
    if not filter_value:
        return True
    normalized_filter = _normalize_token(filter_value)
    for candidate in candidate_values:
        normalized_candidate = _normalize_token(candidate)
        if normalized_filter == normalized_candidate or normalized_filter in normalized_candidate:
            return True
    return False


def _matches_tag_filters(filters: list[str], candidate_tags: list[str]) -> bool:
    if not filters:
        return True
    normalized_tags = [_normalize_token(tag) for tag in candidate_tags]
    for required in filters:
        normalized_required = _normalize_token(required)
        matched = any(
            normalized_required == tag or normalized_required in tag
            for tag in normalized_tags
        )
        if not matched:
            return False
    return True


def _compatibility_status(model_id: str, tags: list[str], architecture: str | None) -> JsonObject:
    normalized_id = _normalize_token(model_id)
    normalized_tags = [_normalize_token(tag) for tag in tags]
    normalized_architecture = _normalize_token(architecture) if architecture else ""

    if normalized_id.startswith("mlx-community/") or any("mlx" == tag or tag.startswith("mlx-") for tag in normalized_tags):
        return {
            "status": "mlx_ready",
            "reason": "Model appears to be MLX-native or explicitly tagged for MLX.",
        }

    convertible_families = {
        "llama",
        "mistral",
        "qwen",
        "phi",
        "gemma",
        "gpt2",
        "falcon",
        "mpt",
    }
    if any(family in normalized_architecture for family in convertible_families):
        return {
            "status": "convertible",
            "reason": "Architecture appears compatible with common MLX conversion workflows.",
        }
    if any("text-generation" in tag or "causal-lm" in tag for tag in normalized_tags):
        return {
            "status": "convertible",
            "reason": "Text-generation model likely convertible to MLX format.",
        }

    return {
        "status": "unsupported",
        "reason": "No strong MLX-ready or convertible compatibility signals were found.",
    }


def _rank_score(item: JsonObject) -> float:
    downloads = item.get("downloads") or 0
    likes = item.get("likes") or 0
    size_class = item.get("parameter_size_class")
    compatibility = item.get("compatibility", {}).get("status")

    score = math.log10(downloads + 1) * 5 + (likes * 0.15)
    if compatibility == "mlx_ready":
        score += 5
    elif compatibility == "convertible":
        score += 2

    if size_class == "small":
        score += 2
    elif size_class == "medium":
        score += 1
    elif size_class == "xlarge":
        score -= 1

    return round(score, 6)


class HFModelClient:
    """Client for Hugging Face model search and inspection."""

    def __init__(self, opener: Opener | None = None, timeout_seconds: int = 15) -> None:
        self._opener = opener or urlopen
        self._timeout_seconds = timeout_seconds

    def search_models(
        self,
        query: str,
        *,
        page: int = 1,
        limit: int = 20,
        size_class: str | None = None,
        tags: list[str] | None = None,
        license_name: str | None = None,
    ) -> JsonObject:
        if page < 1:
            raise ValueError("page must be >= 1")
        if limit < 1:
            raise ValueError("limit must be >= 1")

        offset = (page - 1) * limit
        payload = self._request_json(
            f"{HF_HUB_BASE_URL}/api/models",
            params={
                "search": query,
                "limit": limit,
                "offset": offset,
                "full": "true",
            },
        )
        if isinstance(payload, dict):
            raw_items = payload.get("models", [])
        else:
            raw_items = payload
        if not isinstance(raw_items, list):
            raise APIRequestError("Hugging Face API returned an unexpected model search response shape.")

        normalized_items = [self._normalize_model_item(item) for item in raw_items if isinstance(item, dict)]
        requested_tags = tags or []
        filtered_items = [
            item
            for item in normalized_items
            if (size_class is None or item["parameter_size_class"] == size_class)
            and _matches_filter(license_name, [item["license"]] if item["license"] else [])
            and _matches_tag_filters(requested_tags, item["tags"])
        ]

        ranked_items = sorted(
            filtered_items,
            key=lambda item: (
                item["score"],
                item.get("downloads") or 0,
                item.get("likes") or 0,
                item.get("id") or "",
            ),
            reverse=True,
        )

        return {
            "query": query,
            "page": page,
            "limit": limit,
            "offset": offset,
            "source_count": len(normalized_items),
            "total_returned": len(ranked_items),
            "filters": {
                "size_class": size_class,
                "license": license_name,
                "tags": requested_tags,
            },
            "results": ranked_items,
        }

    def inspect_model(self, model_id: str) -> JsonObject:
        payload = self._request_json(
            f"{HF_HUB_BASE_URL}/api/models/{quote(model_id, safe='/')}",
            params={"full": "true"},
        )
        if not isinstance(payload, dict):
            raise APIRequestError("Hugging Face API returned an unexpected model inspect response shape.")

        normalized = self._normalize_model_item(payload)
        config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
        tokenizer_config = payload.get("tokenizer_config")
        if not isinstance(tokenizer_config, dict):
            tokenizer_config = {}

        tokenizer_info = {
            "class": config.get("tokenizer_class") or tokenizer_config.get("tokenizer_class"),
            "vocab_size": _extract_number(config.get("vocab_size") or tokenizer_config.get("vocab_size")),
            "model_max_length": _extract_number(tokenizer_config.get("model_max_length")),
        }

        return {
            "model_id": normalized["id"] or model_id,
            "summary": normalized["summary"],
            "license": normalized["license"],
            "tags": normalized["tags"],
            "task_tag": normalized["task_tag"],
            "architecture": normalized["architecture"],
            "parameter_count": normalized["parameter_count"],
            "parameter_size_class": normalized["parameter_size_class"],
            "compatibility": normalized["compatibility"],
            "downloads": normalized["downloads"],
            "likes": normalized["likes"],
            "last_modified": normalized["last_modified"],
            "tokenizer": tokenizer_info,
        }

    def _normalize_model_item(self, item: JsonObject) -> JsonObject:
        tags = _dedupe_keep_order(_as_str_list(item.get("tags")))
        card_data = item.get("cardData") if isinstance(item.get("cardData"), dict) else {}
        model_id = str(item.get("id") or item.get("modelId") or item.get("name") or "")
        architecture = _extract_architecture(item, tags)
        parameter_count = _extract_parameter_count(item, tags)
        parameter_size_class = _size_class_for_params(parameter_count)
        compatibility = _compatibility_status(model_id, tags, architecture)

        normalized = {
            "id": model_id,
            "summary": _extract_summary(item, card_data),
            "license": _extract_license(card_data, tags),
            "tags": tags,
            "task_tag": _extract_task_tag(item, tags),
            "architecture": architecture,
            "parameter_count": parameter_count,
            "parameter_size_class": parameter_size_class,
            "compatibility": compatibility,
            "downloads": _extract_number(item.get("downloads")),
            "likes": _extract_number(item.get("likes")),
            "last_modified": item.get("lastModified") or item.get("last_modified"),
        }
        normalized["score"] = _rank_score(normalized)
        return normalized

    def _request_json(self, endpoint: str, *, params: JsonObject | None = None) -> Any:
        encoded_params = urlencode(params or {})
        url = f"{endpoint}?{encoded_params}" if encoded_params else endpoint
        request = Request(
            url=url,
            headers={
                "Accept": "application/json",
                "User-Agent": f"mlx-lab/{__version__}",
            },
        )
        try:
            with self._opener(request, timeout=self._timeout_seconds) as response:
                payload = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace").strip()
            except Exception:
                detail = ""
            message = (
                f"Hugging Face API request failed with HTTP {exc.code} for {url}. "
                "Verify model identifiers and API availability."
            )
            if detail:
                message = f"{message} Response: {detail[:240]}"
            raise APIRequestError(message) from exc
        except URLError as exc:
            raise APIRequestError(
                f"Unable to reach Hugging Face API endpoint {url}. "
                f"Check network connectivity and retry. ({exc.reason})"
            ) from exc
        except TimeoutError as exc:
            raise APIRequestError(
                f"Request timed out while calling Hugging Face API endpoint {url}. "
                "Retry with a stable network connection."
            ) from exc

        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise APIRequestError(
                f"Hugging Face API endpoint {url} returned invalid JSON."
            ) from exc

