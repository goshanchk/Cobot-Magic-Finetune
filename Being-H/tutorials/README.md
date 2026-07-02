# Tutorials

This directory collects practical tutorials and runnable examples built around Being-H models. These tutorials are documentation-oriented examples, not additional Being-H versions. They are meant to show how Being-H models can be adapted, post-trained, evaluated, and deployed on accessible robot platforms and community benchmarks.

## Being-H-EDU

[Being-H-EDU](Being-H-EDU/) is an educational tutorial workspace in this directory. It is built for educational robot workflows on top of the Being-H0.5 VLA codebase, including data preparation, post-training, and local robot inference.

The current public example starts with the SO101 `pick_cube_plate` task:

- SO101 data preparation in LeRobot format
- SO101 action/state mapping into the Being-H unified action space
- Post-training from a Being-H0.5 checkpoint
- Local inference server and SO101 client examples

The SO101 dataset is published at [BeingBeyond/Being-H-EDU_SO101](https://huggingface.co/datasets/BeingBeyond/Being-H-EDU_SO101).

## TODO

Near-term tasks for this tutorial area:

- [ ] Add tutorial support for [BeingBeyond D1](https://github.com/BeingBeyond/Beingbeyond_D1), including data schema, config, training launcher, and deployment notes.
- [ ] Add reusable dataset validation commands for educational robot datasets before post-training.
- [ ] Add additional community benchmark or task-suite examples with minimal configs and reproducible launch commands.
- [ ] Keep robot-specific examples isolated under their own config and example folders as new platforms are added.
