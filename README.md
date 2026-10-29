# RICapBench: Benchmarking Fine-Grained Visual Understanding of Vision-Language Models in Remote Sensing

🤗 Hugging Face (coming soon) · [<img src="img/modelscope.ico" width="20" height="20" alt="ModelScope logo"> ModelScope](https://modelscope.cn/datasets/yyyy12306/RICapBench)

## Overview

RICapBench evaluates fine-grained visual understanding in remote sensing imagery through instance detection, object-level captioning, and spatial relations. This repository provides model-specific fine-tuning configurations, inference scripts, and the RICScore evaluation toolkit.

## Dataset

See the [ModelScope dataset page](https://modelscope.cn/datasets/yyyy12306/RICapBench) for dataset access. A Hugging Face link will be added when available.

For the evaluation data format and prediction examples, see the [RICScore input documentation](RICScore/README.md#input).

## Fine-Tuning Configurations

Use the configurations in [`ms-swift-sft-yaml/`](ms-swift-sft-yaml/) with [ms-swift](https://github.com/modelscope/ms-swift) for fine-tuning.

## Inference

The [`closed_models/`](closed_models/) directory contains API-based inference scripts for closed-source models. Pass the evaluation JSONL path with `--dataset` and set the API environment variables required by the selected script. Use `--help` to view the available options.

## Evaluation

RICScore combines instance detection accuracy, caption quality, and spatial relation F1 into a single score. See the [evaluation guide](RICScore/README.md) for metric definitions, environment setup, input requirements, and evaluation commands.

## Citation

Citation information will be added when publication details are available.

## Acknowledgements

We thank the [ms-swift](https://github.com/modelscope/ms-swift) team for their open-source fine-tuning framework.

## License

This repository is licensed under the [MIT License](LICENSE).
