#!/usr/bin/env bash
set -euo pipefail

# Create a local Python environment for Being-H-EDU.

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup_beingh_env.sh [ENV_DIR]

Default:
  ENV_DIR=.venv

Optional environment variables:
  PYTHON_BIN=python3.10       Python executable used to create the venv.
  INSTALL_FLASH_ATTN=0        Skip flash-attn installation.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ENV_DIR="${1:-.venv}"
if [[ "${ENV_DIR}" == -* ]]; then
  echo "Invalid ENV_DIR: ${ENV_DIR}" >&2
  usage >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

"${PYTHON_BIN}" -m venv "${ENV_DIR}"
source "${ENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install -r "${PROJECT_ROOT}/requirements.txt"

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
    python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
fi

python - <<'PY'
import sys
import torch
import torchvision

print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA runtime:", torch.version.cuda)
PY

cat <<EOF

Environment is ready.
Activate it with:
  source ${ENV_DIR}/bin/activate

Before running Being-H commands, set:
  export PYTHONPATH=${PROJECT_ROOT}:\$PYTHONPATH
EOF
