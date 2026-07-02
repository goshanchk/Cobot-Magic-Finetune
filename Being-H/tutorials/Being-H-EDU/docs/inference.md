# Inference

Use `examples/so101/run_server_so101.sh` to launch the Being-H SO101 inference server.

## Start Server

```bash
MODEL_PATH=/path/to/so101-checkpoint \
bash examples/so101/run_server_so101.sh
```

Default server settings:

```text
SERVER_PORT=8080
DATA_CONFIG_NAME=so101
DATASET_NAME=so101_posttrain
EMBODIMENT_TAG=so101
NUM_INFERENCE_TIMESTEPS=4
USE_MPG=True
ENABLE_RTC=False
```

Override them with environment variables:

```bash
SERVER_PORT=8081 \
NUM_INFERENCE_TIMESTEPS=4 \
MODEL_PATH=/path/to/so101-checkpoint \
bash examples/so101/run_server_so101.sh
```

## Metadata Variant

For the default SO101 training path, the metadata variant is usually inferred automatically. If a checkpoint was trained with a named variant, pass it explicitly:

```bash
METADATA_VARIANT=so101.pick_cube_plate \
MODEL_PATH=/path/to/so101-checkpoint \
bash examples/so101/run_server_so101.sh
```

## Client

`examples/so101/client_so101.py` is a reference client for connecting robot observations to the server. It depends on a local LeRobot SO101 runtime installation and is intentionally kept separate from the training dependencies.
