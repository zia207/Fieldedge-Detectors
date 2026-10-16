#!/usr/bin/env bash
# Rebuild and execute the training notebook from scratch (stages already cached in train_outputs/ are skipped),
# then render the HTML and the code-free PDF.
set -euo pipefail
cd "$(dirname "$0")"
../.venv/bin/python build_training_notebook.py
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=fieldedge --ExecutePreprocessor.timeout=-1 field_boundary_training.ipynb
./render.sh
