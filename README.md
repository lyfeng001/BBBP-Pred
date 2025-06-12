# CS7315 final project
This is the code for the CS7315 final project: Blood-Brain Barrier Penetration (BBBP) Prediction Using Molecular Representation Learning.

# Installation

First, you need to install [uv](https://docs.astral.sh/uv/).

Once uv is installed, install the project dependencies:

```bash
uv sync
```

# Usage

Place the BBBP dataset in the `data/bbbp_data` directory. Then, run the following commands:
- Run the following commands:

    ```base
    uv run model/hyperparameter_search_all_conformers.py
    ```
    for hyperparameter search with multiple conformers.

- Run the following commands:

    ```bash
    uv run model/compare_conformer_strategies.py
    ```    

    for finetuning and testing with either a single conformer or a selection of conformers based on random choice/cluster selection.