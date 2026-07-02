# Being-H-EDU

Being-H-EDU is an educational tutorial workspace built on the Being-H0.5 VLA codebase. It provides a compact path for preparing robot data, post-training a policy, and running local robot inference.

The current public example supports the SO101 `pick_cube_plate` task. The SO101 dataset is published at [BeingBeyond/Being-H-EDU_SO101](https://huggingface.co/datasets/BeingBeyond/Being-H-EDU_SO101).

## Contents

```text
tutorials/Being-H-EDU/
  BeingH/                         # model, dataset, training, and inference code
  configs/
    data_config.py                # SO101 modality/action mapping
    dataset_info.py               # SO101 dataset registry
    posttrain/so101/
      so101_example.yaml          # default post-training YAML
  examples/so101/
    train_so101_beingh.sh         # training launcher
    run_server_so101.sh           # inference server launcher
    client_so101.py               # SO101 client example
    convert_so101_actions_to_delta.py
  tools/so101_processing/         # optional SO101 dataset processing utilities
  docs/                           # setup, data, training, and inference guides
```

## Installation

```bash
cd tutorials/Being-H-EDU
bash scripts/setup_beingh_env.sh .venv
source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
```

If `flash-attn` does not build on your machine, install the environment first with:

```bash
INSTALL_FLASH_ATTN=0 bash scripts/setup_beingh_env.sh .venv
```

Then install a `flash-attn` wheel matching your PyTorch/CUDA setup before training or inference.

## Checkpoints And Data

Set model and data paths through environment variables or command-line arguments. The repository intentionally uses placeholders rather than server-local absolute paths.

Required paths:

- `PRETRAIN_MODEL`: InternVL/Qwen VLM checkpoint, for example `InternVL3_5-2B`.
- `EXPERT_MODEL`: Qwen expert checkpoint, for example `Qwen3-0.6B`.
- `RESUME_PATH`: Being-H checkpoint to resume from.
- `dataset_path_overrides.so101.pick_cube_plate`: local LeRobot-format SO101 dataset path in `configs/posttrain/so101/so101_example.yaml`. See [docs/so101_data_processing.md](docs/so101_data_processing.md) for the Hugging Face dataset path and one-time action conversion step.

The default public dataset key is `so101.pick_cube_plate`. Add your own key such as `so101.my_task` only if you want multiple task-specific registrations.

## Training

Edit `configs/posttrain/so101/so101_example.yaml` and point `so101.pick_cube_plate` to your local dataset, then launch:

```bash
PRETRAIN_MODEL=/path/to/InternVL3_5-2B \
EXPERT_MODEL=/path/to/Qwen3-0.6B \
RESUME_PATH=/path/to/being-h05-checkpoint \
bash examples/so101/train_so101_beingh.sh --gpus 4 --steps 20000
```

See [docs/training.md](docs/training.md) for the main launcher options.

## Inference

```bash
MODEL_PATH=/path/to/so101-checkpoint \
bash examples/so101/run_server_so101.sh
```

See [docs/inference.md](docs/inference.md) for server options and client integration notes.

## Data Processing

SO101 data should be in LeRobot format with delta joint actions before training. The helper scripts under `tools/so101_processing/` can merge datasets and trim static frames. The public SO101 dataset can be used through the path and config flow in [docs/so101_data_processing.md](docs/so101_data_processing.md).

## Documentation

- [Environment setup](docs/environment_setup.md)
- [SO101 quickstart](docs/so101_quickstart.md)
- [SO101 data processing](docs/so101_data_processing.md)
- [Training](docs/training.md)
- [Inference](docs/inference.md)
- [Data configuration](docs/data_configuration.md)
- [Unified action space](docs/unified_action_space.md)

## License

This project is released under Apache-2.0. See [LICENSE](LICENSE).
