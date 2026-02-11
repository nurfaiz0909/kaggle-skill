#!/usr/bin/env bash
# Download datasets and models from Kaggle using the kaggle-cli.
#
# kaggle-cli supports:
#   kaggle datasets download  — download dataset files
#   kaggle datasets files     — list files in a dataset
#   kaggle models get         — download a model (via models instances versions)
#
# Prerequisites:
#   uv pip install kaggle
#   Credentials configured in ~/.kaggle/kaggle.json or env vars
#
# Usage:
#   bash examples/01-download/cli_download.sh

set -euo pipefail

echo "============================================================"
echo "kaggle-cli: Download Dataset"
echo "============================================================"

# List files in the Titanic dataset
echo "--- Listing dataset files ---"
kaggle datasets files titanic/titanic

# Download the entire Titanic dataset to a local directory
echo "--- Downloading dataset ---"
kaggle datasets download titanic/titanic \
    --path ./downloads/titanic \
    --unzip

echo "Dataset downloaded to ./downloads/titanic"
ls -la ./downloads/titanic/

echo ""
echo "============================================================"
echo "kaggle-cli: Download a Specific Dataset File"
echo "============================================================"

# Download only train.csv from the Titanic dataset
kaggle datasets download titanic/titanic \
    --file train.csv \
    --path ./downloads/titanic-single \
    --unzip

echo "Single file downloaded to ./downloads/titanic-single"

echo ""
echo "============================================================"
echo "kaggle-cli: Download Model"
echo "============================================================"

# Download a model using the models command
# Syntax: kaggle models instances versions download <owner>/<model>/<framework>/<variation>/<version>
echo "--- Downloading model ---"
kaggle models instances versions download \
    google/gemma/transformers/2b/1 \
    --path ./downloads/gemma-2b

echo "Model downloaded to ./downloads/gemma-2b"
ls -la ./downloads/gemma-2b/

echo ""
echo "============================================================"
echo "kaggle-cli: Search for Datasets"
echo "============================================================"

# Search for datasets by keyword
kaggle datasets list --search "titanic" --sort-by votes --max-size 10485760

echo ""
echo "============================================================"
echo "kaggle-cli: Search for Models"
echo "============================================================"

# List available models
kaggle models list --search "gemma" --sort-by downloadCount
