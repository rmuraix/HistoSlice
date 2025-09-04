# Repository Guidelines

## Project Structure & Module Organization

- `histoslice/`: core library. Key modules: `_reader.py`, `_backend.py`, `_data.py`; CLI in `cli/`; image ops in `functional/`.
- `tests/`: pytest suite (`*_test.py`), helpers in `tests/_utils.py`, sample assets in `tests/data/`.
- `docs/` + `mkdocs.yml`: user/API docs (MkDocs + mkdocstrings).
- `pyproject.toml`: package metadata, Python `>=3.10`, dev tools; `uv.lock` indicates use of `uv` for env management.

## Build, Test, and Development Commands

- **Environment**: `uv sync` (installs runtime + dev deps).
- **Lint/format**: `uv run ruff check .`
- **Tests**: `uv run pytest -q` (doctests enabled via `pytest.ini`).
- **Docs**: `uv run mkdocs serve` (live preview) or `uv run mkdocs build`.
- **CLI (dev)**: `uv run histoslice --help` (use this while iterating locally).

## Coding Style & Naming Conventions

- **Python style**: 4-space indent, type hints encouraged (project uses annotations throughout).
- **Names**: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- **Imports**: prefer absolute within `histoslice`.
- **Tools**: `ruff` for lint + format; keep formatter clean before committing.

## Testing Guidelines

- **Framework**: `pytest` with doctests (`--doctest-modules`).
- **Layout**: unit tests mirror package paths under `tests/`; name files `*_test.py` and functions `test_*`.
- **Data**: reuse fixtures and test assets in `tests/_utils.py` and `tests/data/` rather than adding large binaries.
- **Run**: `uv run pytest -q`; add targeted tests for new modules and any public bugfix.

## Commit & Pull Request Guidelines

- **Messages**: follow Conventional Commits style seen in history (e.g., `feat:`, `fix:`, `docs:`, `chore:`, `ci:`). Keep imperative, concise, and scoped.
- **PRs**: include a clear description, linked issues (`Closes #123`), test updates, and docs updates when interfaces change. Attach screenshots for CLI/doc output when visual.
- **Quality gate**: ruff passes, tests green, and docs build.

## Security & Configuration Tips

- **External deps**: OpenSlide/PyVips may require system libs. Guard optional backends and provide graceful fallbacks.
- **Data ethics**: do not commit PHI/large slides; use small samples in `tests/data/`.
