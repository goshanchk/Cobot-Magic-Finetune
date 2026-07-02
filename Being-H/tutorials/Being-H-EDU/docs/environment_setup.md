# Environment Setup

Being-H-EDU is validated with Python 3.10 and PyTorch 2.5.x. The repository does not pin a CUDA wheel channel in a separate requirements file; install the PyTorch build that matches your machine.

## Create Environment

```bash
cd tutorials/Being-H-EDU
bash scripts/setup_beingh_env.sh .venv
source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
```

The setup script installs `requirements.txt` and then tries to install `flash-attn==2.7.4.post1` with `--no-build-isolation`.

## Flash Attention

If `flash-attn` fails to build, create the environment without it:

```bash
INSTALL_FLASH_ATTN=0 bash scripts/setup_beingh_env.sh .venv
```

Training and inference import model modules that use FlashAttention. Install a wheel compatible with your PyTorch, Python, CUDA, and GPU driver before running the model.

## Manual Install

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Sanity Check

```bash
python - <<'PY'
import torch
import transformers
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY
```
