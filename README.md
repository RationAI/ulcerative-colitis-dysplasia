# Machine Learning Template

This project provides a machine learning quickstart template using PyTorch Lightning, Hydra, MLflow, and PDM.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd machine-learning
    ```
2.  **Install PDM:**
    Follow the instructions on the [PDM website](https://pdm-project.org/latest/#installation) if you don't have it installed.
3.  **Install dependencies:**
    This command creates a virtual environment and installs all necessary packages defined in [`pyproject.toml`](/Users/matejpekar/Documents/RationAI/machine-learning/pyproject.toml).
    ```bash
    pdm install
    ```
    This also sets up pre-commit hooks for development.

## Configuration

This project uses [Hydra](https://hydra.cc/) for configuration management.

-   Configuration files are located in the [`configs/`](/Users/matejpekar/Documents/RationAI/machine-learning/configs) directory.
-   The main configuration file is [`configs/default.yaml`](/Users/matejpekar/Documents/RationAI/machine-learning/configs/default.yaml).
-   You can override configuration parameters directly from the command line. For example, to change the batch size:
    ```bash
    pdm run train data.batch_size=64
    ```
-   MLflow is configured as the default logger (see [`configs/default.yaml`](/Users/matejpekar/Documents/RationAI/machine-learning/configs/default.yaml)). Ensure your MLflow tracking server is running or configure it accordingly.

## Usage

The project uses PDM scripts defined in [`pyproject.toml`](/Users/matejpekar/Documents/RationAI/machine-learning/pyproject.toml) for common tasks. These scripts execute the main module [`project_name/__main__.py`](/Users/matejpekar/Documents/RationAI/machine-learning/project_name/__main__.py) with different `mode` settings.

-   **Train the model:**
    ```bash
    pdm run train
    ```
    (This runs `python -m project_name mode=fit`)

-   **Validate the model:**
    Requires a checkpoint path to be set in the configuration (e.g., `checkpoint=path/to/your/checkpoint.ckpt`) or passed via the command line.
    ```bash
    pdm run validate checkpoint=path/to/checkpoint.ckpt
    ```
    (This runs `python -m project_name mode=validate ...`)

-   **Test the model:**
    Requires a checkpoint path.
    ```bash
    pdm run test checkpoint=path/to/checkpoint.ckpt
    ```
    (This runs `python -m project_name mode=test ...`)

-   **Run prediction:**
    Requires a checkpoint path.
    ```bash
    pdm run predict checkpoint=path/to/checkpoint.ckpt
    ```
    (This runs `python -m project_name mode=predict ...`)

## Project Structure

```
.
├── configs/              # Hydra configuration files
│   └── default.yaml      # Default configuration
├── project_name/         # Main Python package
│   ├── data/             # Data loading modules (DataModule)
│   │   └── data_module.py
│   ├── __main__.py       # Main script entry point (Hydra app)
│   └── project_name_model.py # PyTorch Lightning model definition
├── pyproject.toml        # Project metadata, dependencies (PDM)
├── README.md             # This file
└── ...                   # Other files (LICENSE, etc.)
```

## Development

-   **Linting and Formatting:** Use Ruff. Run checks with:
    ```bash
    pdm run lint  # Check and fix linting issues
    pdm run format # Format code
    pdm run l     # Run lint, format, and mypy
    ```
-   **Type Checking:** Use MyPy.
    ```bash
    pdm run mypy
    ```
-   **Pre-commit Hooks:** Configured to run Ruff and other checks automatically before each commit. They are installed via `pdm install`.