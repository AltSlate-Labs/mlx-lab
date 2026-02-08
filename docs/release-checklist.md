# Release Checklist

## Scope

Use this checklist for each version release of `mlx-lab`.

## Pre-Release

1. Confirm `.prodman` status:
   - `EP-001` through `EP-007` are `complete`
   - milestone `MS-004` is `complete`
2. Ensure working tree contains only intended release changes.
3. Run full test suite:
   - `uv run python -m unittest discover -s tests -p "test_*.py" -v`
4. Verify packaging build:
   - `uv build --sdist --wheel`
   - Offline/local fallback: `python -m build --sdist --wheel --no-isolation`
5. Verify install smoke:
   - create a clean environment
   - install wheel
   - run `mlx-lab --help`
   - run a minimal local workflow (`data clean` + `train lora --backend simulated`)

## Versioning

1. Bump version in:
   - `pyproject.toml`
   - `src/mlx_lab/_version.py`
2. Update changelog or release notes summary.
3. Commit with conventional message format from `AGENTS.md`.

## Artifact Validation

1. Confirm `dist/` contains both:
   - `mlx_lab-<version>.tar.gz` (sdist)
   - `mlx_lab-<version>-py3-none-any.whl` (wheel)
2. Optional sanity check:
   - `python -m pip install --no-deps --force-reinstall dist/*.whl`
   - `mlx-lab --version`

## Publish

1. Upload artifacts to TestPyPI first (recommended):
   - `python -m twine upload --repository testpypi dist/*`
2. Validate install from TestPyPI in a clean environment.
3. Upload to PyPI:
   - `python -m twine upload dist/*`
4. Tag and publish release notes in VCS hosting.

## Post-Release

1. Verify `pip install mlx-lab` and CLI entrypoint on a clean machine.
2. Archive release notes and commands used.
3. Update roadmap/release status in `.prodman/roadmap.yaml`.
