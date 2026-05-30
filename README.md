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

This repo is the official implementation for the paper [MATA: A Trainable Hierarchical Automaton System for Multi-Agent Visual Reasoning](https://arxiv.org/abs/2601.19204), accepted at ICLR 2026.

MATA formulates visual reasoning as a hierarchical finite-state automaton. A Hyper Agent controls high-level transitions among specialized agents, oneshot reasoning, stepwise reasoning, and answering states, while all agents communicate through shared memory for transparent execution traces.

## Release

- [2026/02] MATA is accepted at ICLR 2026.

## TODOs

We're working on the following TODOs:

- [x] Inference code.
- [ ] MATA-SFT-90K dataset.
- [ ] Training pipeline.
- [ ] Official benchmark evaluation scripts.

## Installation

Clone this repository and install the pixi environment.

```bash
git clone https://github.com/ControlNet/MATA.git
cd MATA
pixi install
```

The default environment targets Linux with CUDA and uses the dependencies specified in [pixi.toml](pixi.toml).

## Environment Variables

Create a local `.env` file from the example file.

```bash
cp .env.example .env
```

Set your OpenAI or OpenAI-compatible endpoint credentials in `.env`.

```bash
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
```

`OPENAI_BASE_URL` is optional when using the official OpenAI endpoint. Local model and tool caches are controlled by `TORCH_HOME` in `.env`.

## Hyper Agent

The Hyper Agent uses a local LLM State Controller to choose state transitions. By default, it uses the Qwen3-4B model, or the future released checkpoint. You can change the model by modifying the `hyper_agent` section in the base config.

```yaml
hyper_agent:
  model_name: "Qwen/Qwen3-4B"
```

## Download Models

Prepare the vision-language tool models used by MATA.

```bash
pixi run download_model
```

## Inference

Run MATA on a single image and query.

```bash
pixi run python scripts/infer_once.py \
  --base_config configs/gqa.yaml \
  --image /path/to/image.jpg \
  --query "How many red cups are on the left table?"
```

## Dataset Inference

Run inference on a dataset split.

```bash
pixi run python scripts/infer_dataset.py \
  --base_config configs/gqa.yaml \
  --data_root /path/to/data \
  --result_folder ./result
```

## Code Structure

```text
src/mata/
  agent/        Hyper Automaton agents and state controller
  execution/    ImagePatch runtime and tool execution helpers
  memory/       Shared memory
  prompt/       Prompt templates
  tool/         Vision-language tool wrappers
  util/         Config, logging, and misc utilities

configs/        Task and model configs
scripts/        Inference entrypoints
```

Key paper terms map to the code as follows:

- Hyper Automaton: [src/mata/agent/mata.py](src/mata/agent/mata.py)
- Hyper Agent and LLM State Controller: [src/mata/agent/hyper_agent.py](src/mata/agent/hyper_agent.py)
- Specialized Agent: [src/mata/agent/specialized](src/mata/agent/specialized)
- Oneshot Reasoner: [src/mata/agent/oneshot](src/mata/agent/oneshot)
- Stepwise Reasoner: [src/mata/agent/stepwise](src/mata/agent/stepwise)
- Answering State: [src/mata/agent/answering](src/mata/agent/answering)
- Shared Memory: [src/mata/memory/shared_memory.py](src/mata/memory/shared_memory.py)

## Citation

If you find this work useful for your research, please consider citing it.

```bibtex
@inproceedings{cai2026mata,
  title={MATA: A Trainable Hierarchical Automaton System for Multi-Agent Visual Reasoning},
  author={Cai, Zhixi and Ke, Fucai and Leo, Kevin and Huang, Sukai and Garcia de la Banda, Maria and Stuckey, Peter J. and Rezatofighi, Hamid},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

## License

This project is released under the [Apache-2.0 License](LICENSE).
