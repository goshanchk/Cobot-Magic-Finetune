# Cobot Magic Fine-Tuning

This repository contains fine-tuning integrations for the Cobot Magic LeRobot dataset. It keeps three model pipelines side by side so the same dataset can be tested with OpenVLA-OFT, NVIDIA Isaac-GR00T N1.7, and LeRobot SmolVLA.

Current training uses only the 14D bimanual joint state/action (`all_arms`). 

## Structure

```text
openvla-oft/       # OpenVLA-OFT integration: LeRobot loader, FSDP/DDP launch commands, TensorBoard logs
Isaac-GR00T/       # Isaac-GR00T integration: Cobot Magic modality config, DeepSpeed launch commands, TensorBoard logs
lerobot/           # LeRobot SmolVLA integration
dataset_description.txt
```

## Read Next

- [OpenVLA-OFT instructions](openvla-oft/README.md)
- [Isaac-GR00T instructions](Isaac-GR00T/README.md)
- [LeRobot SmolVLA instructions](lerobot/README.md)

## Dataset

Expected dataset root in launch examples:

```text
/path/to/cobot_magic_sber
```
