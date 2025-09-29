# HistoSlice - GitHub Copilot Instructions

**ALWAYS** follow these instructions first and only fallback to additional search and context gathering if the information here is incomplete or found to be in error.

HistoSlice is a Python library for preprocessing large medical histological slide images for machine learning. It provides both a CLI tool and Python API for cutting slides into tiles and preprocessing them by removing artifacts, bad tissue areas, etc.

## Working Effectively

### Bootstrap and Environment Setup
- Install `uv` package manager: `pip install uv`
- Set up development environment: `uv sync` -- takes ~7 seconds. NEVER CANCEL.
- **CRITICAL**: Python 3.10+ required. The project uses Python 3.12 in CI.
- **Dependencies**: All required medical imaging libraries (OpenSlide, PyVips, OpenCV) install automatically via `uv sync` - no system packages needed.

### Build and Test Commands
- **Lint**: `uv run ruff check .` -- takes ~0.1 seconds
- **Format check**: `uv run ruff format --check .` -- takes ~0.05 seconds  
- **Format files**: `uv run ruff format .`
- **Tests**: `uv run pytest -q` -- takes ~26 seconds. NEVER CANCEL. Set timeout to 60+ seconds.
  - Runs 90+ tests with doctests enabled
  - Includes ~32 skipped tests (normal for optional backends)
  - May show deprecation warnings (normal)
- **Documentation**: `uv run mkdocs build` -- takes ~2 seconds
- **Documentation serve**: `uv run mkdocs serve` -- for live preview

### CLI Usage
- **Help**: `uv run histoslice --help` -- shows all available options
- **Basic usage**: `uv run histoslice --input './slides/*.tiff' --output ./output --width 512 --overlap 0.5 --max-background 0.5 --thumbnails --metrics`
- **Development CLI**: Use `uv run histoslice` when iterating locally

## Validation

### Manual Testing Requirements
- ALWAYS test CLI functionality after making changes to CLI code
- ALWAYS test Python API after making changes to core library
- **Test CLI**: Run `uv run histoslice --input './tests/data/slide.jpeg' --output /tmp/test-output --width 256 --thumbnails --metrics --num-workers 1`
- **Test Python API**: 
  ```python
  from histoslice import SlideReader
  reader = SlideReader('./tests/data/slide.jpeg')
  threshold, tissue_mask = reader.get_tissue_mask(level=-1)
  tile_coordinates = reader.get_tile_coordinates(tissue_mask, width=256, overlap=0.5, max_background=0.5)
  ```

### CI Requirements
- ALWAYS run `uv run ruff check .` and `uv run ruff format --check .` before committing
- ALWAYS run `uv run pytest -q` before committing 
- The CI (.github/workflows/check.yaml) will fail if linting, formatting, or tests fail

## Common Tasks

### Repository Structure
```
histoslice/                 # Core library
├── __init__.py
├── _reader.py             # Main SlideReader class
├── _backend.py            # Backend management (OpenSlide, PyVips)
├── _data.py               # Data structures (TileCoordinates, etc.)
├── cli/                   # CLI interface
│   ├── _app.py            # Main CLI application
│   ├── _options.py        # CLI option definitions
│   └── _models.py         # Pydantic models for settings
├── functional/            # Image processing functions
└── utils/                 # Utility classes and functions

tests/                     # Test suite
├── *_test.py              # Unit tests
├── _utils.py              # Test utilities
└── data/                  # Test slide images

docs/                      # MkDocs documentation
├── index.md               # Main documentation
└── api/                   # API documentation

.github/workflows/         # CI/CD pipelines
├── check.yaml             # Linting, testing, coverage
├── docs.yaml              # Documentation build/deploy
└── publish.yaml           # PyPI publishing
```

### Key Files to Know
- `pyproject.toml` -- Python package configuration, dependencies, dev tools
- `uv.lock` -- Locked dependency versions (like package-lock.json)
- `mkdocs.yml` -- Documentation configuration
- `AGENTS.md` -- Repository guidelines for agents/contributors
- `tests/data/` -- Sample slide images for testing (JPEG format)

### Development Workflow
1. Make changes to code
2. Run `uv run ruff format .` to format code
3. Run `uv run ruff check .` to check for issues
4. Run `uv run pytest -q` to run tests
5. Test manually with CLI or Python API
6. For docs changes: run `uv run mkdocs serve` to preview

### Medical Imaging Context
- Works with various slide formats: TIFF, SVS, JPEG, CZI
- Typical workflow: Load slide → Detect tissue → Extract tiles → Save with metadata
- **Important**: Test slides in `tests/data/` are small samples - real medical slides can be GB-sized
- **Data Ethics**: Never commit PHI or large medical images - use small samples only

### Common File Locations
- Main entry point: `histoslice/__init__.py`
- CLI entry point: `histoslice/cli/_app.py` 
- Core reader: `histoslice/_reader.py`
- Test utilities: `tests/_utils.py`
- Sample data: `tests/data/slide.jpeg`

### Typical Command Outputs

#### `uv sync` output:
```
Using CPython 3.12.3 interpreter at: /usr/bin/python3.12
Creating virtual environment at: .venv
Resolved 82 packages in 7ms
[downloading and installing packages...]
+ histoslice==0.2.0 (from file:///)
[dependencies installed]
```

#### `uv run pytest -q` output:
```
..sss....ssssssss.ss........ssssss....................ss.ss..........ss..s...................................... [ 91%]
....ssssss                                                                                       [100%]
90 passed, 32 skipped, 49806 warnings in 24.14s
```

#### `uv run histoslice --help` output:
Shows comprehensive CLI help with options for input/output, tile extraction, tissue detection, and saving options.

## Troubleshooting

### Common Issues
- **Import errors**: Run `uv sync` to ensure all dependencies are installed
- **CLI not found**: Use `uv run histoslice` instead of just `histoslice`
- **Test failures**: Check if you're in the project root directory
- **Documentation build issues**: Check for broken links in markdown files

### Performance Notes
- Environment setup: ~7 seconds (full dependency installation)
- Linting: ~0.1 seconds  
- Testing: ~26 seconds (full suite), ~11 seconds (single test file)
- Documentation build: ~2 seconds
- CLI processing: ~2-3 seconds for small test slides
- **TIMEOUT RECOMMENDATIONS**:
  - Environment setup: 15+ minutes timeout (for slow connections)
  - Tests: 120+ seconds timeout (to handle all test scenarios)
  - Documentation build: 60+ seconds timeout
  - CLI operations: 300+ seconds timeout (for larger slides)

### System Requirements
- Python 3.10+ (tested with 3.12.3)
- Linux/macOS/Windows supported
- Medical imaging libraries included in dependencies
- No additional system packages required beyond Python

## CRITICAL REMINDERS

- **NEVER CANCEL** long-running commands:
  - `uv sync` may take up to 10 minutes on slow connections
  - `uv run pytest` may take up to 60 seconds
  - Set timeouts appropriately: environment setup (15+ minutes), tests (120+ seconds)
- **ALWAYS** test CLI and Python API manually after making changes
- **ALWAYS** run linting and formatting before committing
- **ALWAYS** validate that medical imaging functionality works end-to-end