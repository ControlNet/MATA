# MATA: A Trainable Hierarchical Automaton System for Multi-Agent Visual Reasoning

<div align="center">
    <a href="https://github.com/ControlNet/MATA/issues">
        <img src="https://img.shields.io/github/issues/ControlNet/MATA?style=flat-square">
    </a>
    <a href="https://github.com/ControlNet/MATA/network/members">
        <img src="https://img.shields.io/github/forks/ControlNet/MATA?style=flat-square">
    </a>
    <a href="https://github.com/ControlNet/MATA/stargazers">
        <img src="https://img.shields.io/github/stars/ControlNet/MATA?style=flat-square">
    </a>
    <a href="https://arxiv.org/abs/2601.19204">
        <img src="https://img.shields.io/badge/arXiv-2601.19204-b31b1b.svg?style=flat-square">
    </a>
    <a href="https://iclr.cc/virtual/2026/poster/10008264">
        <img src="https://img.shields.io/badge/ICLR-2026-blue.svg?style=flat-square">
    </a>
</div>

**This repo is the official implementation for the paper [MATA: A Trainable Hierarchical Automaton System for Multi-Agent Visual Reasoning](https://arxiv.org/abs/2601.19204), accepted at ICLR 2026.**

MATA (Multi-Agent hierarchical Trainable Automaton) formulates visual reasoning as a hierarchical finite-state automaton. A trainable hyper agent learns high-level transitions across collaborating and competing agents, while each agent executes a small rule-based sub-automaton for reliable micro-control. This design provides transparent execution traces and strong performance on complex visual reasoning benchmarks.

## News

- [2026/02] MATA is accepted at ICLR 2026.

## Abstract

Recent vision-language models have strong perceptual ability but their implicit reasoning is hard to explain and easily generates hallucinations on complex queries. Compositional methods improve interpretability, but most rely on a single agent or hand-crafted pipeline and cannot decide when to collaborate across complementary agents or compete among overlapping ones. We introduce MATA (Multi-Agent hierarchical Trainable Automaton), a multi-agent system presented as a hierarchical finite-state automaton for visual reasoning whose top-level transitions are chosen by a trainable hyper agent. Each agent corresponds to a state in the hyper automaton, and runs a small rule-based sub-automaton for reliable micro-control. All agents read and write a shared memory, yielding transparent execution history. To supervise the hyper agent's transition policy, we build transition-trajectory trees and transform to memory-to-next-state pairs, forming the MATA-SFT-90K dataset for supervised finetuning (SFT). The finetuned LLM as the transition policy understands the query and the capacity of agents, and it can efficiently choose the optimal agent to solve the task. Across multiple visual reasoning benchmarks, MATA achieves the state-of-the-art results compared with monolithic and compositional baselines.

## Planned Release

- [ ] Training and inference
- [ ] MATA-SFT-90K dataset
- [ ] Evaluation

## Citation

If you find this work useful for your research, please consider citing it.

```bibtex
@article{cai2026mata,
  title={MATA: A Trainable Hierarchical Automaton System for Multi-Agent Visual Reasoning},
  author={Cai, Zhixi and Ke, Fucai and Leo, Kevin and Huang, Sukai and de la Banda, Maria Garcia and Stuckey, Peter J and Rezatofighi, Hamid},
  journal={arXiv preprint arXiv:2601.19204},
  year={2026}
}
```
