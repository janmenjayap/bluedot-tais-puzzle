#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-bluedot-impact-puzzle-1-py311}"
PYTHON_VERSION="3.11"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create env if it doesn't exist
if ! conda env list | grep -qE "(^|[[:space:]])${ENV_NAME}([[:space:]]|$)"; then
  conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
fi

# Keep later activated sessions isolated from packages installed under ~/.local.
conda env config vars set -n "$ENV_NAME" PYTHONNOUSERSITE=1 >/dev/null

# Install dependencies into the env
PYTHONNOUSERSITE=1 conda run -n "$ENV_NAME" \
  python -m pip install -r "$REPO_DIR/requirements.txt"
PYTHONNOUSERSITE=1 conda run -n "$ENV_NAME" python -m pip check
echo "Setup complete. Activate with: conda activate $ENV_NAME"
