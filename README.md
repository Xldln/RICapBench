# RICapBench: A Fine-Grained Visual Understanding Benchmark Dataset for Vision-Language Models in Remote Sensing

[🤗 Hugging Face](https://huggingface.co/datasets/yubinux/RICapBench) · [<img src="img/modelscope.ico" width="20" height="20" alt="ModelScope logo"> ModelScope](https://modelscope.cn/datasets/yyyy12306/RICapBench)

## 🌏 Overview

RICapBench evaluates fine-grained visual understanding in remote sensing imagery through instance detection, object-level captioning, and spatial relations. This repository provides model-specific fine-tuning configurations, inference scripts, and the RICScore evaluation toolkit.

## 🗂️ Dataset

The dataset is available on [Hugging Face](https://huggingface.co/datasets/yubinux/RICapBench) and [ModelScope](https://modelscope.cn/datasets/yyyy12306/RICapBench).

For the evaluation data format and prediction examples, see the [RICScore input documentation](RICScore/README.md#input).

## 🛠️ Fine-Tuning Configurations

Use the configurations in [`ms-swift-sft-yaml/`](ms-swift-sft-yaml/) with [ms-swift](https://github.com/modelscope/ms-swift) for fine-tuning.

Clone and install ms-swift from source. Run the following commands from the RICapBench repository root:

```bash
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .
cd ..
```

Choose the YAML file for your model and replace the `/path/to/...` placeholders with your local model, dataset, image, cache, and special-token paths. Adjust `ENV.NPROC_PER_NODE` and the batch size for your available GPUs. The configurations use FlashAttention and log to TensorBoard and Weights & Biases; install the corresponding dependencies and configure logging before training.

For example, fine-tune Qwen3-VL-2B-Instruct from the RICapBench repository root:

```bash
swift sft ms-swift-sft-yaml/Qwen3-VL-2B-Instruct.yaml
```

See the [ms-swift documentation](https://swift.readthedocs.io/en/latest/) for environment requirements and model-specific dependencies.

## 🤖 Inference

The [`closed_models/`](closed_models/) directory contains API-based inference scripts for closed-source models. Pass the evaluation JSONL path with `--dataset` and set the API environment variables required by the selected script. Use `--help` to view the available options.

## 📊 Evaluation

RICScore combines instance detection accuracy, caption quality, and spatial relation F1 into a single score. See the [evaluation guide](RICScore/README.md) for metric definitions, environment setup, input requirements, and evaluation commands.

## Citation

Citation information will be added when publication details are available.

## Acknowledgements

We thank the [ms-swift](https://github.com/modelscope/ms-swift) team for their open-source fine-tuning framework.

## 📜 License

This repository is licensed under the [MIT License](LICENSE).
