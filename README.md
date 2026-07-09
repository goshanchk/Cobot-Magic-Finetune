# Cobot Magic Fine-Tuning

This repository contains fine-tuning integrations for the Cobot Magic LeRobot dataset. It keeps four model pipelines side by side so the same dataset can be tested with OpenVLA-OFT, NVIDIA Isaac-GR00T N1.7, LeRobot SmolVLA, and Being-H05.

Current training uses only the 14D bimanual joint state/action (`all_arms`). 

## Structure

```text
openvla-oft/       # OpenVLA-OFT integration: LeRobot loader, FSDP/DDP launch commands, TensorBoard logs
Isaac-GR00T/       # Isaac-GR00T integration: Cobot Magic modality config, DeepSpeed launch commands, 
lerobot/           # LeRobot SmolVLA integration
Being-H/           # Being-H05 integration: Cobot Magic modality config, FSDP launch commands
dataset_description.txt
```

## Read Next

- [OpenVLA-OFT instructions](openvla-oft/README.md)
- [Isaac-GR00T instructions](Isaac-GR00T/README.md)
- [LeRobot SmolVLA instructions](lerobot/README.md)
- [Being-H05 instructions](Being-H/README.md)

## Model Pipelines

### OpenVLA-OFT

OpenVLA-OFT fine-tuning on the Cobot Magic dataset with OpenVLA-OFT's robot integration scripts. See [openvla-oft/README.md](openvla-oft/README.md).

### Isaac-GR00T

NVIDIA Isaac-GR00T N1.7 fine-tuning with Cobot Magic modality configuration and DeepSpeed launch commands. See [Isaac-GR00T/README.md](Isaac-GR00T/README.md).

### LeRobot SmolVLA

LeRobot SmolVLA training against the same Cobot Magic LeRobot dataset. See [lerobot/README.md](lerobot/README.md).

### Being-H05

Being-H05 fine-tuning uses the Cobot Magic 14D bimanual joint state/action mapping into Being-H's 200D unified action space. See [Being-H/README.md](Being-H/README.md).

## Dataset

Expected dataset root in launch examples:

```text
/path/to/cobot_magic_sber
```
