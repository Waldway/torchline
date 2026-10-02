# Project Setup (Zed + uv)

The code is in normal `.py` files. Each file is split into cells with `# %%`. You run the cells one by one in Zed, like a notebook.

## What you need

- [Zed](https://zed.dev)
- [uv](https://docs.astral.sh/uv/) (a fast tool that installs Python and packages)

Install uv on macOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup (once)

Run these in the project folder:

```bash
uv sync
uv run python -m ipykernel install --user --name torchline
```

- `uv sync` creates a `.venv` folder and installs all packages from `pyproject.toml`.
- The second command registers the project as a _kernel_ (the program that runs your code), so Zed can find it.

## Run the code in Zed

1. Open the project folder in Zed.
2. Open a `.py` file.
3. Press **Cmd+Shift+P**, run **repl: sessions**, and pick `torchline`.
4. Put the cursor in a cell and press **Cmd+Shift+Enter** (Linux/Windows: **Ctrl+Shift+Enter**).

The output appears under the cell. Plots appear there too.

## Add a package

```bash
uv add package-name
```

Then restart the kernel in Zed (**repl: sessions** → restart).

## Run a whole file without Zed

```bash
uv run python training.py
```

## Files and folders

Run everything from this `python/` folder, because the paths are relative to it.

- `training.py`: trains the saturation model and exports weights for Rust
- `inference.py`: tube simulation model; exports `../models/model.json`
- `saturation.py`: hand-written saturation and delay effects
- `envelope_follower.py`: envelope follower experiments
- `ir_wizard.py`: CLI that slices recordings into impulse responses (`uv run python ir_wizard.py --help`)
- `audio/`: put your `.wav` files here (not committed to git)
- `../models/`: shared with the Rust plugin, which embeds `model.json` and `weights_causal.json` at build time
