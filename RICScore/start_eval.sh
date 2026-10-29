#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Edit these defaults or override them with environment variables.
# Relative data and output paths are resolved from the current working directory.
PYTHON="${PYTHON:-python}"
EVAL_DIR="${EVAL_DIR:-outputs/evaluation}"
IMAGE_ROOT="${IMAGE_ROOT:-data/RICapBench}"
EVAL_DATASET_JSONL="${EVAL_DATASET_JSONL:-${IMAGE_ROOT}/evaluation.jsonl}"
GT_RELATIONS="${GT_RELATIONS:-${IMAGE_ROOT}/evaluation-spatial-gt-relations.jsonl}"
PREDICTIONS="${PREDICTIONS:-${EVAL_DIR}/infer_predictions.jsonl}"
SPICE_MODEL_DIR="${SPICE_MODEL_DIR:-${SCRIPT_DIR}/third_party/stanford}"
SPATIAL_PARSER_DIR="${SPATIAL_PARSER_DIR:-${SCRIPT_DIR}/spatial_relations}"
SPICE_CACHE_DIR="${SPICE_CACHE_DIR:-${SCRIPT_DIR}/.cache/spice}"

for file in "$PREDICTIONS" "$EVAL_DATASET_JSONL" "$GT_RELATIONS"; do
  [[ -f "$file" ]] || { echo "File not found: $file. Configure the paths in start_eval.sh or through environment variables." >&2; exit 1; }
done
[[ -d "$IMAGE_ROOT" ]] || { echo "Image root not found: $IMAGE_ROOT" >&2; exit 1; }

# Convert structured predictions and attach the original ground truth.
"$PYTHON" "$SCRIPT_DIR/convert_predictions.py" \
  "$PREDICTIONS" "${EVAL_DIR}/ricscore_input.json" \
  --eval-dataset "$EVAL_DATASET_JSONL" \
  --image-root "$IMAGE_ROOT"

# Score the converted input using the fixed spatial-relation ground truth.
"$PYTHON" "$SCRIPT_DIR/ricscore.py" \
  "${EVAL_DIR}/ricscore_input.json" \
  --geometry rbox \
  --gt-relations "$GT_RELATIONS" \
  --spatial-parser-dir "$SPATIAL_PARSER_DIR" \
  --spice-model-dir "$SPICE_MODEL_DIR" \
  --spice-cache-dir "$SPICE_CACHE_DIR" \
  --output "${EVAL_DIR}/ricscore_result.json"

# Display the key metrics without changing the values stored in the report.
"$PYTHON" - "${EVAL_DIR}/ricscore_result.json" <<'SUMMARY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
report = json.loads(source.read_text(encoding="utf-8"))
percent_metrics = (
    "RICScore", "DetA", "CapA", "RelCap_F1_matched", "RelCap_F1_all",
    "METEOR", "CIDEr_D_normalized", "SPICE_OA", "SPICE_All",
)
def display(title, metrics):
    print(f"\n{title}")
    for name in percent_metrics:
        value = metrics[name]
        formatted = "N/A" if value is None else f"{value:.2%}"
        label = "CIDEr_D / 10" if name == "CIDEr_D_normalized" else name
        print(f"  {label:<24} {formatted}")
    value = metrics["CIDEr_D"]
    formatted = "N/A" if value is None else f"{value:.2f}"
    print(f"  {'CIDEr_D (raw)':<24} {formatted}")
for geometry, metrics in report["metrics"].items():
    display(f"Evaluation results ({geometry})", metrics)
    for subset, values in metrics.get("subsets", {}).items():
        display(f"  {subset} ({geometry})", values)
print(f"\nFull report: {source}")
SUMMARY
