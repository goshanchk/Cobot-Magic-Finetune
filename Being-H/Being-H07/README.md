# Being-H0.7: A Latent World-Action Model from Egocentric Videos

<div align="center">

[![Blog](https://img.shields.io/badge/Blog-Being--H07-green)](https://research.beingbeyond.com/being-h07)
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/pdf/2605.00078)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](../LICENSE)

</div>

Being-H0.7 is BeingBeyond's flagship **WAM** model, built as a **latent world-action model** that learns to organize future-relevant interaction structure in latent space and generate actions without explicit future-image rollout at test time.

## News

- **[2026-04-14]**: We publish **Being-H0.7**. Read the [blog](https://research.beingbeyond.com/being-h07) and [paper](https://arxiv.org/abs/2605.00078) for the latest overview.

## Overview

Being-H0.7 is trained from large-scale human egocentric video together with robot demonstrations.
The core idea is to insert a compact latent reasoning interface between multimodal perception and action generation, then shape this interface with a dual-branch training scheme:

- A deployable **prior branch** that only sees current instruction, observation history, and state.
- A training-only **posterior branch** that additionally observes future frames.
- Latent alignment that distills future-aware structure into the deployable branch.

## Citation

```bibtex
@article{beingbeyond2026beingh07,
  title={Being-H0. 7: A Latent World-Action Model from Egocentric Videos},
  author={Luo, Hao and Zhang, Wanpeng and Feng, Yicheng and Zheng, Sipeng and Xu, Haiweng and Xu, Chaoyi and Xi, Ziheng and Fu, Yuhui and Lu, Zongqing},
  journal={arXiv preprint arXiv:2605.00078},
  year={2026}
}
```
