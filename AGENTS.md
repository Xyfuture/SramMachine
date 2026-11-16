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

## PU 内部带宽设计

- 每个 PU 拥有 DRAM（HBM）和 Flash 两种带宽资源，模拟器需要为每个 PU 分别实例化这两种资源。
- 每个算子（Op）可以仅使用 DRAM、仅使用 Flash，或同时使用两者。计算权重（weight）加载时间时，必须根据该算子实际使用的带宽来源计算。
- 同时使用 DRAM 和 Flash 时，需要自动分配权重加载任务：按照两者的带宽比例分配数据量，使两者的加载时间相同。设权重总数据量为 `W`，DRAM 和 Flash 带宽分别为 `B_dram` 和 `B_flash`，则分配的数据量分别为 `W * B_dram / (B_dram + B_flash)` 和 `W * B_flash / (B_dram + B_flash)`，并行加载时间为 `W / (B_dram + B_flash)`。
- DRAM 带宽保持固定；Flash 带宽可随场景和算子变化。计算加载时间和任务分配比例时，应使用该算子指定的 Flash 带宽；未指定时，使用默认值。

## Pareto 图绘制

绘制Pareto 图的时候，横轴控制为 tokens/s/user，纵轴控制为 tokens/s/chip ， 采用线性坐标。


## 一些要求:

- 默认情况下，请运行xfold模式。明确说明的情况下，可以跑普通的模式
- Flash的带宽使用csv表格中提供的数据


## Coding Style & Naming Conventions

Use four spaces, standard Python formatting, type hints, and docstrings for public classes and functions. Follow `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. Keep model and operator identifiers consistent with existing model cards and mapping keys. No formatter or linter is configured, so keep imports ordered and changes readable by inspection.

## Testing Guidelines

Add focused `unittest.TestCase` methods under `srammachine/tests/` with filenames matching `test_*.py`. Use descriptive `test_...` names and `subTest` for model, batch, or strategy matrices. Run the full discovery command before submitting changes; mapping or simulator changes should include regression coverage for timing, resource, or command-graph invariants.

## Commit & Pull Request Guidelines

Recent commits use short, imperative, lower-case summaries such as `fix some bugs` and `change attn split logic`. Keep commits focused and use a similarly concise subject. Pull requests should explain the behavioral change, identify affected models or strategies, list test commands and results, and include representative generated artifacts or screenshots when output formats or traces change.
