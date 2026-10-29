"""Convert ``swift infer`` JSONL output into RICScore input JSON."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from PIL import Image


from spatial_relations.dataset import parse_answer, _image_value, _dataset_answer, image_path_and_id


def _resolve_image(value: str, image_root: Path):
    path, image_id = image_path_and_id(value, image_root)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return path, image_id


def _to_pixels(objects, width, height, *, ground_truth):
    converted = []
    for index, item in enumerate(objects):
        cx, cy, box_width, box_height, angle = item["rbox"]
        result = {
            "id": f"{'g' if ground_truth else 'p'}{index}",
            "label": item["label"],
            "rbox": [cx * width / 1000, cy * height / 1000,
                     box_width * width / 1000, box_height * height / 1000, angle],
        }
        if ground_truth:
            result["captions"] = [item["caption"]] if item["caption"] else []
        else:
            result["caption"] = item["caption"]
        converted.append(result)
    return converted


def convert_row(row, image_root: Path, *, ground_truth):
    image_path, image_id = _resolve_image(_image_value(row), image_root)
    with Image.open(image_path) as image:
        width, height = image.size
    truths, _, _ = parse_answer(_dataset_answer(ground_truth), strict=True)
    predictions, format_valid, invalid_count = parse_answer(row.get("response"))
    return {
        "image_id": ground_truth.get("image_id", image_id),
        "prediction_format_valid": format_valid,
        "invalid_prediction_count": invalid_count,
        "ground_truth": _to_pixels(truths, width, height, ground_truth=True),
        "predictions": _to_pixels(predictions, width, height, ground_truth=False),
    }


def _read_jsonl(source):
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Each JSONL row must be an object")
            except ValueError as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error
            yield row


def convert_file(source: Path, target: Path, image_root: Path, *, eval_dataset: Path):
    if target.resolve() in {source.resolve(), eval_dataset.resolve()}:
        raise ValueError("Output must not overwrite predictions or evaluation GT")
    truths, image_ids = {}, set()
    for row in _read_jsonl(eval_dataset):
        path, image_id = _resolve_image(_image_value(row), image_root)
        identifier = row.get("image_id", image_id)
        if isinstance(identifier, bool) or not isinstance(identifier, (str, int)) or not str(identifier):
            raise ValueError("Evaluation image IDs must be non-empty strings or integers")
        if path in truths or str(identifier) in image_ids:
            raise ValueError(f"Duplicate evaluation image: {identifier}")
        parse_answer(_dataset_answer(row), strict=True)
        truths[path] = row
        image_ids.add(str(identifier))
    if not truths:
        raise ValueError("Evaluation dataset is empty")
    records, seen = [], set()
    for row in _read_jsonl(source):
        path, _ = _resolve_image(_image_value(row), image_root)
        if path not in truths:
            raise ValueError(f"Unexpected prediction image: {path}")
        if path in seen:
            raise ValueError(f"Duplicate prediction image: {path}")
        if "image_id" in row:
            _, default_id = _resolve_image(_image_value(truths[path]), image_root)
            if str(row["image_id"]) != str(truths[path].get("image_id", default_id)):
                raise ValueError(f"Prediction image_id disagrees with evaluation dataset: {path}")
        records.append(convert_row(row, image_root, ground_truth=truths[path]))
        seen.add(path)
    missing = truths.keys() - seen
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} evaluation images: {sorted(map(str, missing))[:5]}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=target.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps({"images": records}, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(records), sum(record["prediction_format_valid"] for record in records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL written by swift infer")
    parser.add_argument("output", type=Path, help="RICScore input JSON")
    parser.add_argument("--eval-dataset", type=Path, required=True,
                        help="Original evaluation JSONL; supplies GT and the complete image set")
    parser.add_argument("--image-root", type=Path, default=Path("."),
                        help="Root directory for relative image paths (default: current directory)")
    args = parser.parse_args()
    try:
        count, valid = convert_file(args.input, args.output, args.image_root,
                                    eval_dataset=args.eval_dataset)
        print(f"Converted {count} images; canonical prediction format: {valid}/{count}")
    except (ValueError, OSError) as error:
        parser.exit(1, f"RICScore conversion failed: {error}\n")


if __name__ == "__main__":
    main()
