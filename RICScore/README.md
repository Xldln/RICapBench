# RICScore

Evaluation of instance detection, captions, and spatial relations on RICapBench (v0.1.0).

## Metrics

```text
DetA = TP_det / (TP_det + FP_det + FN_det)
CapA = mean_pair[(METEOR + CIDEr-D / 10 + SPICE-OA) / 3]
RelCap_F1_matched = 2 TP_rel / (2 TP_rel + FP_rel + FN_rel)
RICScore = (DetA × CapA × RelCap_F1_matched)^(1/3)
```

- Detection uses class-agnostic one-to-one matching, with IoU >= 0.5 by default. Matching prioritizes pair count, total IoU, then total CapA.
- CapA averages matched pairs with reference captions. CIDEr-D uses all caption-annotated GT instances as its fixed IDF corpus. SPICE-OA pools object/attribute fact counts per pair using maximum one-to-one semantic matching; it differs from official SPICE All.
- Relation F1 pools spatial-fact counts over matched instances. Full-scope `RelCap_F1_all` is reported separately and does not enter RICScore. Relations are extracted from text, without checking image pixels.
- No detection matches gives RICScore zero. Empty evaluations, no caption-annotated matches, or empty relation sets on both sides of all matched instances give unavailable scores (`null`), with the no-detection-match rule taking priority.

## Setup

Requirements: Python 3.10+, JDK 11+ with `jdk.compiler`, and `curl`.
For example, with Conda:

```bash
conda create -n ricscore python=3.12 -y
conda activate ricscore
conda install -c conda-forge openjdk=11 -y
bash build_eval_env.sh
```

Run commands from the repository root. The setup script installs Python dependencies and downloads CoreNLP 3.6.0 and SLF4J JARs to `third_party/stanford/`; it does not install Java. To use another resource directory, set `SPICE_MODEL_DIR` when running both setup and evaluation.

## Input

Provide the original evaluation JSONL, its images, and model prediction JSONL. Data, images, and model JARs are not bundled. Predictions must cover exactly the evaluation image set, without missing, extra, or duplicate images. GT comes from the original evaluation dataset.

A prediction row has an `image` path (relative to `IMAGE_ROOT`, or absolute) and a `response` string:

```json
{"image": "example.jpg", "response": "<rbox><500><500><200><100><angle><30></rbox><ref>car</ref><caption>A car is beside a road.</caption>"}
```

Boxes use `[cx, cy, width, height, angle]`: coordinates are on a 0–1000 scale; angles are in degrees in `[0, 180)`. Multiple objects are concatenated in the response. Prediction `<ref>` labels are optional; GT labels are required. Use `<null>` for no objects. Empty prediction captions are retained, and recognizable invalid box attempts count as detection FP.

The evaluation JSONL uses the same image fields, with GT in the final assistant turn of `messages` or `conversations` (or in `response`/`labels`).

## Usage


### Run evaluation

Edit the variables at the top of `start_eval.sh`, or override them:

```bash
IMAGE_ROOT=data/RICapBench \
EVAL_DATASET_JSONL=data/RICapBench/evaluation.jsonl \
GT_RELATIONS=data/RICapBench/evaluation-spatial-gt-relations.jsonl \
PREDICTIONS=outputs/my_model/infer_predictions.jsonl \
EVAL_DIR=outputs/my_model \
SPICE_MODEL_DIR=third_party/stanford \
bash start_eval.sh
```
