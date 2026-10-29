"""Evaluate Qwen on RICapBench with one independent API request per image."""

import argparse
import base64
import json
import os
import re
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path("/path/to/RICapBench/evaluation.jsonl")
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/qwen_3.8_max/infer_predictions.jsonl"
DEFAULT_MODEL = "qwen3.8-max-0902"
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
CATEGORIES = (
    "airplane", "airport", "baseball field", "basketball court", "bicycle",
    "boat", "bridge", "bus", "car", "cargo truck", "chimney", "dam",
    "dry cargo ship", "dump truck", "engineering ship", "excavator",
    "expressway service area", "expressway toll station", "fishing boat",
    "football field", "golf course", "ground track field", "harbor",
    "intersection", "liquid cargo ship", "motorboat", "motorcycle",
    "other ship", "other vehicle", "overpass", "passenger ship", "person",
    "roundabout", "ship", "small car", "stadium", "storage tank",
    "tennis court", "tractor", "traffic light", "traffic sign", "trailer",
    "train station", "tricycle", "truck", "truck tractor", "tugboat", "van",
    "vehicle", "warship", "windmill",
)

SYSTEM_PROMPT = """You perform object-level dense captioning for aerial and satellite images.

Detect all visible instances belonging to the allowed categories. For each
instance, predict a tight oriented bounding box, its category, and one concise
English caption.

Allowed categories:
""" + ", ".join(CATEGORIES) + """

Detection rules:
- Inspect the full image systematically, including small instances, dense
  groups, and image borders. Do not merge a row or cluster into one box.
- Use exactly one category name from the allowed list for each instance.
- Copy category names exactly, including spaces. Do not introduce hyphens
  or substitute synonyms; use "ground track field", not "ground-track field".
- Report each instance once. Describe individual objects separately rather
  than merging multiple instances into one detection.
- Include recognizable instances that are partially visible at image borders.
- Do not report objects outside the allowed categories.

Oriented bounding box convention:
- The image origin is the top-left corner; x increases rightward and y downward.
- Represent each box by its center (cx, cy), long-side length w, short-side
  length h, and angle theta, with w >= h.
- Normalize cx and w by the original image width, and cy and h by the original
  image height, multiplying each by 1000 and rounding to integers.
- Center coordinates must be in [0, 1000]. Side lengths must be positive;
  a long side extending diagonally can exceed 1000 after normalization.
- Theta is the clockwise angle in degrees from the positive x-axis to the
  long side, rounded to an integer in [0, 179].
- At theta=0 the long side is horizontal; at theta=90 it is vertical.
- A long side sloping down to the right has theta=45; a long side sloping
  down to the left has theta=135. The angle describes the box axis, not
  the object's heading.
- Fit the box to the visible extent of the instance.

Caption rules:
- Write one concise English sentence about the corresponding instance.
- Include discriminative visible appearance and the exact category name used
  in <ref> as a contiguous phrase in the caption's subject. Preserve the category's
  wording and word order; place color and other modifiers before the full category
  name, or describe attributes later in the sentence. For example, for "small car",
  write "A white small car ..." rather than "A small white car ...".
- Make the target instance the grammatical subject. When clearly supported,
  describe its spatial relationship to identifiable nearby objects or physical
  landmarks. Name the reference object explicitly rather than using vague
  phrases such as "another one" or "the surrounding area".
- Image position may supplement a description, but should not replace a visible
  relationship to a physical reference object. If no clear reference object is
  visible, describe appearance and image position without inventing a relationship.
- Describe only visually supported attributes and relationships.
- Do not mention coordinates, boxes, or the annotation process.

Output rules:
- For each instance, output exactly:
  <rbox><cx><cy><w><h><angle><theta></rbox><ref>category</ref><caption>sentence</caption>
- Replace the placeholders with predicted values and text.
- Enclose every numeric value in its own complete <...> pair. Keep >< between
  consecutive numeric values: write <258><790>, never <258<790>.
- Concatenate the instance records without numbering or additional text.
- If no allowed-category instance is visible, output exactly <null>.
- Do not output Markdown, explanations, or a scene-level summary.
- Before returning the answer, check that every record is complete, all side
  lengths are positive with w >= h, and all angles are integers from 0 to 179.

Fictional format example (not an object to copy into the answer):
<rbox><500><300><180><60><angle><45></rbox><ref>car</ref><caption>A white car is positioned diagonally beside the road near a grassy area.</caption>
"""

RECORD_PATTERN = re.compile(
    r"<rbox><(\d+)><(\d+)><(\d+)><(\d+)><angle><(\d+)></rbox>"
    r"(?:<ref>([^<>]*)</ref>)?<caption>(.*?)</caption>",
    re.DOTALL,
)


def load_samples(dataset, limit=None):
    samples = []
    seen = set()
    root = dataset.parent.resolve()
    with dataset.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            image = row["image"]
            path = (root / image).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"Line {line_number}: missing image or path outside dataset: {image}")
            if image in seen:
                raise ValueError(f"Line {line_number}: duplicate image: {image}")
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                raise ValueError(f"Unsupported image type: {image}")
            conversations = row["conversations"]
            if len(conversations) != 2 or [item["from"] for item in conversations] != ["human", "assistant"]:
                raise ValueError(f"Line {line_number}: expected one human/assistant pair")
            question, labels = (item["value"] for item in conversations)
            if not isinstance(question, str) or not question.strip() or not isinstance(labels, str):
                raise ValueError(f"Line {line_number}: invalid instruction or labels")
            samples.append({"image": image, "path": path, "question": question, "labels": labels})
            seen.add(image)
            if limit is not None and len(samples) >= limit:
                break
    return samples


def response_format_error(response):
    # Diagnose canonical syntax; the scorer can still retain usable partial records.
    text = response.split("<|im_end|>", 1)[0].strip()
    if text == "<null>":
        return None
    matches = list(RECORD_PATTERN.finditer(text))
    if not matches or RECORD_PATTERN.sub("", text).strip():
        return "Expected complete rbox/ref/caption records or <null>"
    for match in matches:
        cx, cy, width, height, theta = map(int, match.groups()[:5])
        if not (0 <= cx <= 1000 and 0 <= cy <= 1000 and width > 0 and height > 0 and 0 <= theta < 180):
            return "Invalid box center, side lengths, or angle"
    return None


def response_prompt_warnings(response):
    warnings = []
    for index, match in enumerate(RECORD_PATTERN.finditer(response.split("<|im_end|>", 1)[0]), 1):
        if match[6] not in CATEGORIES:
            warnings.append(f"Record {index}: missing or unknown category: {match[6]}")
        if not match[7].strip():
            warnings.append(f"Record {index}: empty caption")
    return warnings


def predict(client, sample, dataset, config):
    record = {
        "response": "", "labels": sample["labels"], "logprobs": None,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["question"]},
        ],
        "images": [{"bytes": None, "path": sample["image"]}],
        "dataset": str(dataset), "generation_config": config,
    }
    try:
        mime = "image/png" if sample["path"].suffix.lower() == ".png" else "image/jpeg"
        encoded = base64.b64encode(sample["path"].read_bytes()).decode("ascii")
        # Every call contains only the fixed system prompt and this image's instruction.
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                {"type": "text", "text": sample["question"]},
            ]},
        ]
        extra_body = {"enable_thinking": config["enable_thinking"]}
        if "vl_high_resolution_images" in config:
            extra_body["vl_high_resolution_images"] = config["vl_high_resolution_images"]
        completion = client.chat.completions.create(
            model=config["model"], messages=messages,
            temperature=config["temperature"], max_tokens=config["max_tokens"],
            extra_body=extra_body,
        )
        choice = completion.choices[0]
        record["response"] = choice.message.content or ""
        record["messages"].append({"role": "assistant", "content": record["response"]})
        record["finish_reason"] = choice.finish_reason
        record["model"] = completion.model
        record["usage"] = completion.usage.model_dump() if completion.usage else None
        record["format_error"] = response_format_error(record["response"])
        record["prompt_warnings"] = response_prompt_warnings(record["response"])
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def load_existing(output, samples, dataset, config):
    records = {}
    if not output.exists():
        return records
    by_image = {sample["image"]: sample for sample in samples}
    with output.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Only an unfinished final append can be discarded safely.
                if not line.endswith("\n") and not stream.read():
                    break
                raise ValueError(f"Invalid existing JSONL at line {number}: {output}") from None
            image = record["images"][0]["path"]
            sample = by_image.get(image)
            if (sample is None or record.get("dataset") != str(dataset)
                    or record.get("generation_config") != config
                    or record.get("labels") != sample["labels"]
                    or record.get("messages", [])[:2] != [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": sample["question"]},
                    ]):
                raise ValueError("Existing output uses different data or settings; choose another --output")
            records[image] = record
    return records


def save_ordered(output, samples, records):
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for sample in samples:
            if sample["image"] in records:
                stream.write(json.dumps(records[sample["image"]], ensure_ascii=False) + "\n")
    temporary.replace(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=65536)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable model thinking (requires endpoint support)")
    parser.add_argument("--high-resolution", action=argparse.BooleanOptionalAction, default=None,
                        help="Set vl_high_resolution_images; omitted uses the endpoint default")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if min(args.workers, args.max_tokens, args.timeout) <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("workers, max-tokens, timeout, and limit must be positive")
    dataset, output = args.dataset.resolve(), args.output.resolve()
    if output == dataset:
        parser.error("output must not overwrite the evaluation dataset")
    samples = load_samples(dataset, args.limit)
    print(f"Validated {len(samples)} samples and image paths", flush=True)
    if args.validate_only:
        return 0

    if not QWEN_BASE_URL.strip() or not QWEN_API_KEY.strip():
        parser.error("Set the QWEN_BASE_URL and QWEN_API_KEY environment variables")
    base_url, api_key = QWEN_BASE_URL, QWEN_API_KEY
    config = {
        "model": args.model, "base_url": base_url, "temperature": 0,
        "max_tokens": args.max_tokens, "enable_thinking": args.thinking,
    }
    if args.high_resolution is not None:
        config["vl_high_resolution_images"] = args.high_resolution
    records = load_existing(output, samples, dataset, config) if args.resume else {}
    pending = [sample for sample in samples
               if sample["image"] not in records or "error" in records[sample["image"]]]
    try:
        from openai import OpenAI
    except ImportError:
        parser.error("Missing dependency: install it with python -m pip install openai")
    try:
        from tqdm import tqdm
    except ImportError:
        parser.error("Missing dependency: install it with python -m pip install tqdm")

    output.parent.mkdir(parents=True, exist_ok=True)
    save_ordered(output, samples, records)
    print(f"model={args.model} pending={len(pending)} workers={args.workers} output={output}", flush=True)
    # SDK retries cover transient API errors; model output is never corrected or resampled.
    with OpenAI(api_key=api_key, base_url=base_url, timeout=args.timeout, max_retries=2) as client:
        try:
            with output.open("a", encoding="utf-8") as stream, \
                    ThreadPoolExecutor(max_workers=args.workers) as executor, \
                    tqdm(total=len(samples), initial=len(samples) - len(pending),
                         desc="Inference", unit="image", dynamic_ncols=True) as progress:
                remaining = iter(pending)
                futures = {executor.submit(predict, client, sample, dataset, config): sample
                           for sample in islice(remaining, args.workers)}
                while futures:
                    completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in completed:
                        sample = futures.pop(future)
                        record = future.result()
                        records[sample["image"]] = record
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush()
                        status = record.get("error") or record.get("format_error") or record.get("finish_reason")
                        progress.set_postfix_str(str(status), refresh=False)
                        progress.update(1)
                        following = next(remaining, None)
                        if following is not None:
                            futures[executor.submit(predict, client, following, dataset, config)] = following
        finally:
            save_ordered(output, samples, records)

    failures = sum("error" in record for record in records.values())
    invalid = sum(bool(record.get("format_error")) for record in records.values())
    warned = sum(bool(record.get("prompt_warnings")) for record in records.values())
    truncated = sum(record.get("finish_reason") == "length" for record in records.values())
    print(f"Saved {len(records)} records; API failures={failures}, format errors={invalid}, "
          f"prompt warnings={warned}, truncated={truncated}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
