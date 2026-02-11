"""Download datasets and models from Kaggle using kagglehub.

kagglehub supports:
  - dataset_download() — download any public or private dataset
  - model_download()   — download any public or private model
  - dataset_load()     — download + load into Pandas/Polars/HuggingFace

Usage:
    python examples/01-download/kagglehub_download.py
"""

import kagglehub


def download_dataset() -> str:
    """Download a dataset using kagglehub.

    Returns the local path where the dataset was saved.
    """
    # Download the latest version of the Titanic dataset
    path = kagglehub.dataset_download("titanic/titanic")
    print(f"Dataset downloaded to: {path}")

    # Download a specific version (version is encoded in the handle string)
    path_v1 = kagglehub.dataset_download("titanic/titanic/versions/1")
    print(f"Dataset v1 downloaded to: {path_v1}")

    return path


def download_model() -> str:
    """Download a model using kagglehub.

    Returns the local path where the model was saved.
    """
    # Download the latest version of a model
    path = kagglehub.model_download("google/gemma/transformers/2b")
    print(f"Model downloaded to: {path}")

    # Download a specific version (version is part of the handle)
    path_v1 = kagglehub.model_download("google/gemma/transformers/2b/1")
    print(f"Model v1 downloaded to: {path_v1}")

    # Download a specific file from the model
    path_file = kagglehub.model_download(
        "google/gemma/transformers/2b",
        path="config.json",
    )
    print(f"Model config downloaded to: {path_file}")

    return path


def load_dataset_with_adapter():
    """Download and load a dataset directly into a DataFrame.

    kagglehub provides adapters for:
      - KaggleDatasetAdapter.PANDAS
      - KaggleDatasetAdapter.POLARS
      - KaggleDatasetAdapter.HUGGING_FACE
    """
    from kagglehub import KaggleDatasetAdapter

    # Load directly into a pandas DataFrame
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "titanic/titanic",
        sql_query="SELECT * FROM train",  # or specify a file path
    )
    print(f"Loaded DataFrame with {len(df)} rows and {len(df.columns)} columns")
    print(df.head())
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("kagglehub: Download Dataset")
    print("=" * 60)
    download_dataset()

    print()
    print("=" * 60)
    print("kagglehub: Download Model")
    print("=" * 60)
    download_model()

    print()
    print("=" * 60)
    print("kagglehub: Load Dataset with Adapter")
    print("=" * 60)
    load_dataset_with_adapter()
