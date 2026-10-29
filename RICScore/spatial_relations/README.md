# Spatial relations

Extract `[subject, predicate, object]` facts from English instance captions for RICScore. Subjects use the allowed category vocabulary; objects use physical landmarks. Image-position references are excluded. Extraction uses CoreNLP 3.6.0 and text rules, without checking image pixels.

## Setup

Requires Python 3.10+ and JDK 11+ with `jdk.compiler`. From the repository root, run `bash build_eval_env.sh` to install Python dependencies and download Java resources. See the [RICScore setup](../README.md#setup).

## Command line

From the repository root:

```bash
python -m spatial_relations data/RICapBench/evaluation.jsonl \
  data/RICapBench/evaluation-spatial-gt-relations.jsonl \
  --image-root data/RICapBench \
  --model-dir third_party/stanford \
  --cache-dir .cache/spatial
```

Input accepts evaluation JSONL or instance JSONL with `caption` and optional `label`. Evaluation input produces one output row per GT instance, including instances with no relations. Use the same `--image-root` as prediction conversion.

The output includes relations and the parser version required by RICScore. GT compatibility checks use only the parser version; older rule-version and policy-fingerprint fields are ignored. Optional controls include `--library-dir` for additional Java dependencies, `--batch-size`, `--threads`, and `--timeout`; use `python -m spatial_relations --help` for all options.

## Python

```python
from spatial_relations import parse_captions

facts = parse_captions(
    ["A white small car is beside a road."],
    model_dir="third_party/stanford",
    labels=["small car"],
)
assert facts == [[["small car", "beside", "road"]]]
```

If supplied, `labels` must contain one string per caption. Without a label, the subject is resolved from the opening subject phrase. Unresolved subjects produce no facts. `cache_dir` enables persistent caching.

For standalone installation, run `pip install ./spatial_relations`; the CLI is `drc-spatial-relations`.
