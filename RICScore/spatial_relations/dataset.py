"""Canonical instance format shared by prediction conversion and relation GT extraction."""
import math
import re
from pathlib import Path


ITEM_PATTERN = re.compile(
    r"<rbox><(?P<cx>\d+)><(?P<cy>\d+)><(?P<w>\d+)><(?P<h>\d+)>"
    r"<angle><(?P<angle>\d+)></rbox>(?:<ref>(?P<label>[^<>]*)</ref>)?"
    r"<caption>(?P<caption>.*?)</caption>",
    re.DOTALL,
)
BOX_PATTERN = re.compile(
    r"<rbox><(?P<cx>\d+)><(?P<cy>\d+)><(?P<w>\d+)><(?P<h>\d+)>"
    r"<angle><(?P<angle>\d+)></rbox>(?:<ref>(?P<label>[^<>]*)</ref>)?"
)
CAPTION_PATTERN = re.compile(r"<caption>(.*?)</caption>", re.DOTALL)


def parse_answer(text: str, *, strict: bool = False):
    """Return usable objects, canonical-format validity, and invalid-object FP count."""
    if not isinstance(text, str):
        raise ValueError("response and labels must be strings")
    answer = text.split("<|im_end|>", 1)[0].strip()
    if answer == "<null>":
        return [], True, 0

    # ponytail: count only <rbox>-marked attempts; add another grammar if the output format changes.
    starts = [match.start() for match in re.finditer(r"<rbox>", answer)]
    objects, invalid_count = [], 0
    valid = bool(starts) and not answer[:starts[0]].strip()
    for start, end in zip(starts, starts[1:] + [len(answer)]):
        block = answer[start:end].strip()
        match = BOX_PATTERN.match(block)
        if match is None:
            invalid_count += 1
            valid = False
            continue
        box = [float(match.group(name)) for name in ("cx", "cy", "w", "h", "angle")]
        label = (match.group("label") or "").strip()
        item_valid = (all(math.isfinite(value) for value in box)
                      and 0 <= box[0] <= 1000 and 0 <= box[1] <= 1000
                      and box[2] > 0 and box[3] > 0 and 0 <= box[4] < 180)
        valid &= item_valid and ITEM_PATTERN.fullmatch(block) is not None
        if item_valid:
            caption = CAPTION_PATTERN.match(block[match.end():])
            objects.append({"rbox": box, "label": label,
                            "caption": caption.group(1).strip() if caption else ""})
        else:
            invalid_count += 1
    if strict and (not valid or any(not obj["label"] for obj in objects)):
        raise ValueError(f"Invalid canonical RIC answer: {answer[:240]!r}")
    return objects, bool(valid), invalid_count


def _image_value(row):
    value = row.get("image", row.get("images"))
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("Each RICScore row must contain exactly one image")
        value = value[0]
    if isinstance(value, dict):
        value = value.get("path")
    if not isinstance(value, str) or not value:
        raise ValueError("Could not find an image path in 'image' or 'images'")
    return value


def image_path_and_id(value, image_root):
    """Use one image-ID convention for relation GT and prediction conversion."""
    root = Path(image_root).resolve()
    path = (root / value).resolve()
    image_id = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()
    return path, image_id


def _dataset_answer(row):
    if "messages" in row:
        messages = row["messages"]
        if (not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict)
                or messages[-1].get("role") != "assistant"):
            raise ValueError("Evaluation messages must end with the ground-truth assistant answer")
        return messages[-1].get("content")
    if "conversations" in row:
        messages = row["conversations"]
        if (not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict)
                or messages[-1].get("from") not in {"assistant", "gpt"}):
            raise ValueError("Evaluation conversations must end with the ground-truth assistant answer")
        return messages[-1].get("value")
    return row.get("response", row.get("labels"))


