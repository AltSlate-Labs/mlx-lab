"""Hugging Face dataset discovery and inspection client."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from ._version import __version__

JsonObject = dict[str, Any]
Opener = Callable[..., Any]

HF_HUB_BASE_URL = "https://huggingface.co"
HF_DATASETS_SERVER_BASE_URL = "https://datasets-server.huggingface.co"


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
    results: list[str] = []
    normalized_prefixes = tuple(prefix.lower() for prefix in prefixes)
    for tag in tags:
        if ":" not in tag:
            continue
        key, value = tag.split(":", 1)
        if key.lower() in normalized_prefixes and value:
            results.append(value)
    return results


def _matches_filter(filter_value: str | None, candidate_values: list[str]) -> bool:
    if not filter_value:
        return True
    normalized_filter = _normalize_token(filter_value)
    for candidate in candidate_values:
        normalized_candidate = _normalize_token(candidate)
        if normalized_filter == normalized_candidate or normalized_filter in normalized_candidate:
            return True
    return False


def _extract_summary(card_data: JsonObject, fallback: str = "") -> str:
    for key in ("summary", "description", "dataset_summary", "pretty_name"):
        value = card_data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback.strip()


def _extract_languages(card_data: JsonObject, tags: list[str]) -> list[str]:
    languages = _as_str_list(card_data.get("language")) + _as_str_list(card_data.get("languages"))
    languages.extend(_extract_tag_values(tags, ("language",)))
    return _dedupe_keep_order([lang for lang in languages if lang])


def _extract_license(card_data: JsonObject, tags: list[str]) -> str | None:
    license_values = _as_str_list(card_data.get("license"))
    license_values.extend(_extract_tag_values(tags, ("license",)))
    for value in license_values:
        if value:
            return value
    return None


def _extract_tasks(card_data: JsonObject, tags: list[str]) -> list[str]:
    task_values: list[str] = []
    task_values.extend(_as_str_list(card_data.get("task_categories")))
    task_values.extend(_as_str_list(card_data.get("task_ids")))
    task_values.extend(_extract_tag_values(tags, ("task_categories", "task_ids", "task")))
    return _dedupe_keep_order([task for task in task_values if task])


def _extract_num(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _extract_features(node: Any) -> Any:
    if isinstance(node, dict):
        features = node.get("features")
        if features not in (None, {}, []):
            return features
        for value in node.values():
            extracted = _extract_features(value)
            if extracted is not None:
                return extracted
    if isinstance(node, list):
        for value in node:
            extracted = _extract_features(value)
            if extracted is not None:
                return extracted
    return None


def _collect_split_rows(node: Any, rows: list[dict[str, Any]]) -> None:
    if isinstance(node, dict):
        split_name = node.get("split") or node.get("name")
        if isinstance(split_name, str) and split_name:
            rows.append(
                {
                    "name": split_name,
                    "num_rows": _extract_num(node.get("num_examples") or node.get("num_rows")),
                }
            )
        for value in node.values():
            _collect_split_rows(value, rows)
        return
    if isinstance(node, list):
        for item in node:
            _collect_split_rows(item, rows)


def _dedupe_splits(splits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for split in splits:
        name = split.get("name")
        if not isinstance(name, str):
            continue
        normalized = _normalize_token(name)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(split)
    return deduped


class HFDatasetClient:
    """Client for Hugging Face dataset search and inspection."""

    def __init__(self, opener: Opener | None = None, timeout_seconds: int = 15) -> None:
        self._opener = opener or urlopen
        self._timeout_seconds = timeout_seconds

    def search_datasets(
        self,
        query: str,
        *,
        page: int = 1,
        limit: int = 20,
        language: str | None = None,
        task: str | None = None,
        license_name: str | None = None,
    ) -> JsonObject:
        if page < 1:
            raise ValueError("page must be >= 1")
        if limit < 1:
            raise ValueError("limit must be >= 1")

        offset = (page - 1) * limit
        payload = self._request_json(
            f"{HF_HUB_BASE_URL}/api/datasets",
            params={
                "search": query,
                "limit": limit,
                "offset": offset,
                "full": "true",
            },
        )

        if isinstance(payload, dict):
            raw_results = payload.get("datasets", [])
        else:
            raw_results = payload
        if not isinstance(raw_results, list):
            raise APIRequestError("Hugging Face API returned an unexpected search response shape.")

        results = [self._normalize_dataset_item(item) for item in raw_results if isinstance(item, dict)]
        filtered_results = [
            item
            for item in results
            if _matches_filter(language, item["languages"])
            and _matches_filter(task, item["task_tags"])
            and _matches_filter(license_name, [item["license"]] if item["license"] else [])
        ]

        return {
            "query": query,
            "page": page,
            "limit": limit,
            "offset": offset,
            "source_count": len(results),
            "total_returned": len(filtered_results),
            "filters": {
                "language": language,
                "task": task,
                "license": license_name,
            },
            "results": filtered_results,
        }

    def inspect_dataset(self, dataset_id: str) -> JsonObject:
        metadata = self._request_json(
            f"{HF_HUB_BASE_URL}/api/datasets/{quote(dataset_id, safe='/')}",
            params={"full": "true"},
        )
        if not isinstance(metadata, dict):
            raise APIRequestError("Hugging Face API returned an unexpected inspect response shape.")

        normalized = self._normalize_dataset_item(metadata)
        card_data = metadata.get("cardData") if isinstance(metadata.get("cardData"), dict) else {}

        warnings: list[str] = []
        splits = _dedupe_splits(_extract_splits_from_metadata(metadata))
        if not splits:
            try:
                splits = self._fetch_splits(dataset_id)
            except APIRequestError as exc:
                warnings.append(str(exc))

        feature_schema = _extract_features_from_metadata(metadata)
        if feature_schema is None:
            try:
                feature_schema = self._fetch_feature_schema(dataset_id)
            except APIRequestError as exc:
                warnings.append(str(exc))

        result: JsonObject = {
            "dataset_id": normalized["id"] or dataset_id,
            "summary": _extract_summary(card_data, fallback=normalized["summary"]),
            "license": normalized["license"],
            "languages": normalized["languages"],
            "task_tags": normalized["task_tags"],
            "downloads": normalized["downloads"],
            "likes": normalized["likes"],
            "last_modified": normalized["last_modified"],
            "splits": splits,
            "feature_schema": feature_schema,
        }
        if warnings:
            result["warnings"] = warnings
        return result

    def _fetch_splits(self, dataset_id: str) -> list[dict[str, Any]]:
        payload = self._request_json(
            f"{HF_DATASETS_SERVER_BASE_URL}/splits",
            params={"dataset": dataset_id},
        )
        split_rows: list[dict[str, Any]] = []
        _collect_split_rows(payload, split_rows)
        return _dedupe_splits(split_rows)

    def _fetch_feature_schema(self, dataset_id: str) -> Any:
        payload = self._request_json(
            f"{HF_DATASETS_SERVER_BASE_URL}/info",
            params={"dataset": dataset_id},
        )
        return _extract_features(payload)

    def _normalize_dataset_item(self, item: JsonObject) -> JsonObject:
        card_data = item.get("cardData") if isinstance(item.get("cardData"), dict) else {}
        tags = _as_str_list(item.get("tags"))
        summary_fallback = ""
        for key in ("description", "readme", "prettyName"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                summary_fallback = value.strip()
                break
        return {
            "id": str(item.get("id") or item.get("name") or ""),
            "summary": _extract_summary(card_data, fallback=summary_fallback),
            "languages": _extract_languages(card_data, tags),
            "license": _extract_license(card_data, tags),
            "task_tags": _extract_tasks(card_data, tags),
            "downloads": _extract_num(item.get("downloads")),
            "likes": _extract_num(item.get("likes")),
            "last_modified": item.get("lastModified") or item.get("last_modified"),
        }

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
                "Verify dataset identifiers and API availability."
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


def _extract_splits_from_metadata(metadata: JsonObject) -> list[dict[str, Any]]:
    split_rows: list[dict[str, Any]] = []
    for key in ("dataset_info", "datasetInfo", "cardData"):
        value = metadata.get(key)
        _collect_split_rows(value, split_rows)
    return split_rows


def _extract_features_from_metadata(metadata: JsonObject) -> Any:
    for key in ("dataset_info", "datasetInfo", "cardData"):
        value = metadata.get(key)
        features = _extract_features(value)
        if features is not None:
            return features
    return None

