# Repository Guidelines

## Project Structure & Module Organization

The Python package lives in `srammachine/`. Core areas are:

- `hardware/`, `inference/`, and `mapping/` define model and accelerator configuration.
- `pipetree/` builds execution trees; `commands/` represents command graphs.
- `simulator/` executes graphs and exports timing or Perfetto results.
- `search/` contains SplitTree and simulated-annealing optimization.
- `frontend/` contains operator definitions; `model_cards/` stores model JSON data.
- `script/` contains runnable experiments and their documentation.
- `tests/` contains regression tests, currently using `unittest`.

Run commands from the repository root so package imports and relative output paths resolve correctly. Generated results belong in ignored directories such as `/test result/` or `/best split tree result/`.

## Build, Test, and Development Commands

There is no separate build system. Use the repository-local virtual environment at `.venv/` to run the project from the repository root. Invoke Python as `.venv/bin/python` (and install or sync dependencies into this environment), rather than relying on whichever Python happens to be active in the shell:

```bash
.venv/bin/python -m unittest discover -s srammachine/tests -p 'test_*.py'
.venv/bin/python -m srammachine.script.run_sa --help
.venv/bin/python -m srammachine.script.run_sa --models deepseek-v3 --mtp off --batch-sizes 256 --rounds 10 --input-sequence-length 20000 --output-sequence-length 600 --moe-strategy tp
```

The first command runs all regression tests. The second validates the experiment CLI; the third runs a small simulated-annealing case. See `srammachine/script/README.md` for full experiment options and output formats.

## Coding Style & Naming Conventions

Use four spaces, standard Python formatting, type hints, and docstrings for public classes and functions. Follow `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. Keep model and operator identifiers consistent with existing model cards and mapping keys. No formatter or linter is configured, so keep imports ordered and changes readable by inspection.

## Testing Guidelines

Add focused `unittest.TestCase` methods under `srammachine/tests/` with filenames matching `test_*.py`. Use descriptive `test_...` names and `subTest` for model, batch, or strategy matrices. Run the full discovery command before submitting changes; mapping or simulator changes should include regression coverage for timing, resource, or command-graph invariants.

## Commit & Pull Request Guidelines

Recent commits use short, imperative, lower-case summaries such as `fix some bugs` and `change attn split logic`. Keep commits focused and use a similarly concise subject. Pull requests should explain the behavioral change, identify affected models or strategies, list test commands and results, and include representative generated artifacts or screenshots when output formats or traces change.
