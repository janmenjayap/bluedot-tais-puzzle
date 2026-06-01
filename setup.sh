#!/usr/bin/env bash
set -euo pipefail
ENV_NAME="bluedot-impact-puzzle-1"
# Create env if it doesn't exist
if ! conda env list | grep -qE "(^|[[:space:]])${ENV_NAME}([[:space:]]|$)"; then
  conda create -y -n "$ENV_NAME" python=3.14.5
fi
# Install dependencies into the env
conda run -n "$ENV_NAME" pip install -r requirements.txt
echo "Setup complete. Activate with: conda activate $ENV_NAME"
