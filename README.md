# Machine Learning Template

This project provides a machine learning quickstart template using PyTorch Lightning, Hydra, MLflow, and uv.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd machine-learning
    ```
2.  **Install uv:**
    Follow the instructions on the [uv website](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it installed.
3.  **Install dependencies:**
    This command creates a virtual environment and installs all necessary packages defined in [`pyproject.toml`](pyproject.toml).
    ```bash
    uv sync
    ```

## Configuration

This project uses [Hydra](https://hydra.cc/) for configuration management.

-   Configuration files are located in the [`configs/`](configs) directory.
-   The main configuration file is [`configs/project_name.yaml`](configs/default.yaml).
-   You can override configuration parameters directly from the command line. For example, to change the batch size:
    ```bash
    uv run python +m <project_name> mode=fit data.batch_size=64
    ```
-   MLflow is configured as the default logger (see [`configs/default.yaml`](configs/default.yaml)). Ensure your MLflow tracking server is running or configure it accordingly.

## Usage

-   **Train the model:**
    ```bash
    uv run python +m <project_name> mode=fit
    ```

-   **Validate the model:**
    Requires a checkpoint path to be set in the configuration (e.g., `checkpoint=path/to/your/checkpoint.ckpt`) or passed via the command line.
    ```bash
    uv run python +m <project_name> mode=validate checkpoint=path/to/checkpoint.ckpt
    ```

-   **Test the model:**
    Requires a checkpoint path.
    ```bash
    uv run python +m <project_name> mode=test checkpoint=path/to/checkpoint.ckpt
    ```

-   **Run prediction:**
    Requires a checkpoint path.
    ```bash
    uv run python +m <project_name> mode=predict checkpoint=path/to/checkpoint.ckpt
    ```


## Linting, Formatting and Type Checking:

```bash
uvx ruff check  # Check and fix linting issues
uvx ruff format # Format code
uvx mypy .     # Run mypy
```