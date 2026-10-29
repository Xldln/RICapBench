"""Independent caption-to-spatial-triples parser; Python standard library only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile

PARSER_VERSION = "drc-spatial-0.1.0"
PACKAGE_DIR = Path(__file__).resolve().parent
SPATIAL_PREDICATES = frozenset(
    line.split("\t", 1)[0] for line in (PACKAGE_DIR / "spatial_predicates.tsv").read_text().splitlines()
    if line and not line.startswith("#"))

CATEGORY_PATH = PACKAGE_DIR.parent / "semantic_similarity_categories.json"
if not CATEGORY_PATH.is_file():
    CATEGORY_PATH = PACKAGE_DIR / "semantic_similarity_categories.json"
OBJECT_PATH = PACKAGE_DIR / "object_landmarks.txt"
FORBIDDEN_OBJECTS = frozenset("area areas screen screens scene scenes image images picture pictures frame frames".split())
PREDICATE_ALIASES = dict(
    line.split("\t")[1:] for line in (PACKAGE_DIR / "fact_equivalence.tsv").read_text().splitlines()
    if line.startswith("alias\t"))


def normalize_predicate(value):
    value = " ".join(value.lower().split())
    return PREDICATE_ALIASES.get(value, value)


def category_names(path=CATEGORY_PATH):
    groups = json.loads(Path(path).read_text(encoding="utf-8"))["groups"]
    if not isinstance(groups, dict) or any(not isinstance(v, list) for v in groups.values()):
        raise ValueError("Category groups must map names to lists")
    names = set(groups)
    for values in groups.values():
        names.update(values)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Category names must be nonempty strings")
    return frozenset(name.lower() for name in names)


def object_names():
    """Allowed physical noun lemmas and their English plural forms."""
    lemmas = {line for line in OBJECT_PATH.read_text(encoding="utf-8").splitlines()
              if line and not line.startswith("#")}
    names = set(lemmas)
    for word in lemmas:
        if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            names.add(word[:-1] + "ies")
        elif word.endswith(("s", "x", "z", "ch", "sh")):
            names.add(word + "es")
        else:
            names.add(word + "s")
    for singular, plural in {"person": "people", "wharf": "wharves"}.items():
        if singular in lemmas:
            names.add(plural)
    return frozenset(names - FORBIDDEN_OBJECTS)


def _cache_fingerprint(categories_path=CATEGORY_PATH):
    """Invalidate cached parses when implementation or resources change."""
    paths = [Path(__file__), PACKAGE_DIR / "SpatialRelationParser.java",
             PACKAGE_DIR / "spatial_predicates.tsv", PACKAGE_DIR / "fact_equivalence.tsv",
             PACKAGE_DIR / "RICRelationScorer.java", OBJECT_PATH, Path(categories_path)]
    digest = hashlib.sha256(PARSER_VERSION.encode())
    for path in paths:
        digest.update(hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).digest())
    return digest.hexdigest()


def resolve_subject(caption, label="", *, categories=None):
    """Resolve the first nominal subject, never a category in a later PP/clause."""
    categories = category_names() if categories is None else categories
    text = " ".join(caption.split()).lower()
    offset = 0
    # Introductory location/time clauses precede the actual first subject: "On the right, a car...".
    intro = re.match(r"\s*(?:in|on|at|near|above|below|under|during|beside|positioned|located|situated)\b[^.!?]*?,\s*(?=(?:a|an|the|this|that|these|those)\b)", text)
    if intro:
        offset = intro.end()
    opening = text[offset:]
    # Commas can separate adjectives before the subject: "a slender, pale bridge".
    boundary = re.search(r"\b(?:is|are|was|were|has|have|with|without|in|on|at|near|beside|among|between|behind|above|below|along|under|over|to|from|of|sits?|stands?|lies?|appears?|surround(?:s|ed|ing)?|border(?:s|ed|ing)?|touch(?:es|ed|ing)?|intersect(?:s|ed|ing)?|overlap(?:s|ped|ping)?|cross(?:es|ed|ing)?|encircle(?:s|d)?|flank(?:s|ed|ing)?|adjoin(?:s|ed|ing)?|abut(?:s|ted|ting)?|straddle(?:s|d)?|spans?|faces?|facing|carries|carrying|rides?|riding|walks?|walking|travels?|moving|while)\b|[.;!?]", opening)
    prefix = opening[:boundary.start()] if boundary else opening
    if re.search(r"\b(?:no(?!-)|neither|without)\b", prefix):
        return None
    hits = []
    for name in categories:
        spelling = name.replace("_", " ")
        plural = "people" if spelling == "person" else spelling + ("es" if spelling.endswith("bus") else "s")
        match = re.search(r"(?<![a-z])(?:" + re.escape(plural) + "|" + re.escape(spelling) + r")(?![a-z])", prefix)
        if match:
            hits.append((match.start(), -len(spelling), name, match.group()))
    if not hits:
        return None
    hits.sort()
    first = hits[0]
    label = label.lower().strip()
    preferred = [hit for hit in hits if hit[2] == label and hit[0] <= first[0] + len(first[3])]
    chosen = preferred[0] if preferred else first
    return {"subject": chosen[2], "surface": chosen[3], "start": offset + chosen[0]}


def parse_captions(captions, model_dir, *, library_dir=None, batch_size=1000,
                   cache_dir=None, threads=8, timeout=3600, progress=False, labels=None,
                   categories_path=CATEGORY_PATH):
    """Return one sorted list of [subject, spatial predicate, reference] per caption."""
    captions = list(captions)
    if batch_size <= 0 or threads <= 0 or timeout <= 0:
        raise ValueError("batch_size, threads, and timeout must be positive")
    if any(not isinstance(caption, str) for caption in captions):
        raise ValueError("Captions must be strings")
    captions = [" ".join(caption.split()) for caption in captions]
    labels = [""] * len(captions) if labels is None else list(labels)
    if len(labels) != len(captions) or any(not isinstance(label, str) for label in labels):
        raise ValueError("labels must contain one string per caption")
    categories = category_names(categories_path)
    allowed_objects = object_names()
    subjects = [resolve_subject(caption, label, categories=categories) for caption, label in zip(captions, labels)]
    requests = [(caption, subject["subject"], subject["surface"], subject["start"]) if subject else (caption, "", "", 0)
                for caption, subject in zip(captions, subjects)]
    if not captions:
        return []
    model_dir = Path(model_dir).resolve()
    for suffix in ("", "-models"):
        if not (model_dir / f"stanford-corenlp-3.6.0{suffix}.jar").is_file():
            raise FileNotFoundError(f"Missing Stanford CoreNLP 3.6.0{suffix} jar in {model_dir}")
    directories = list(dict.fromkeys([model_dir, Path(library_dir).resolve() if library_dir else model_dir]))
    resources = [(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                 for directory in directories for p in sorted(directory.glob("*.jar"))]
    fingerprint = hashlib.sha256(_cache_fingerprint(categories_path).encode()
                                 + json.dumps(resources).encode()).hexdigest()
    connection = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(cache_dir / "spatial.sqlite3")
        connection.execute("CREATE TABLE IF NOT EXISTS parses (key TEXT PRIMARY KEY, result TEXT NOT NULL)")
    results = {request: [] for request in requests if not request[1]}
    def key(text):
        return hashlib.sha256((fingerprint + json.dumps(text)).encode()).hexdigest()
    try:
        for text in dict.fromkeys(requests):
            if text in results:
                continue
            cached = connection.execute("SELECT result FROM parses WHERE key=?", (key(text),)).fetchone() if connection else None
            if cached:
                results[text] = json.loads(cached[0])
        pending = [text for text in dict.fromkeys(requests) if text not in results]
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            if progress:
                print(f"Spatial parsing {min(start + batch_size, len(pending))}/{len(pending)} uncached captions", file=sys.stderr, flush=True)
            with tempfile.TemporaryDirectory(prefix="drc-spatial-") as directory:
                source, target = Path(directory) / "captions.txt", Path(directory) / "relations.jsonl"
                source.write_text("\n".join(f"{subject}\t{surface}\t{start}\t{caption}" for caption, subject, surface, start in batch) + "\n", encoding="utf-8")
                result = subprocess.run([
                    "java", "-Xmx4G", "-cp", os.pathsep.join(str(p / "*") for p in directories),
                    str(PACKAGE_DIR / "SpatialRelationParser.java"), str(source), str(target),
                    str(PACKAGE_DIR / "spatial_predicates.tsv"), str(threads), str(OBJECT_PATH),
                    str(PACKAGE_DIR / "fact_equivalence.tsv")],
                    capture_output=True, text=True, timeout=timeout)
                if result.returncode:
                    raise RuntimeError(f"Spatial parser failed ({result.returncode}): {result.stderr[-6000:]}")
                rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
            if len(rows) != len(batch):
                raise RuntimeError("Spatial parser returned the wrong number of captions")
            for text, triples in zip(batch, rows):
                if not isinstance(triples, list) or any(
                    not isinstance(t, list) or len(t) != 3 or any(not isinstance(w, str) or not w.strip() for w in t)
                    or t[1] not in SPATIAL_PREDICATES
                    or any(member not in allowed_objects for member in t[2].split(" & ")) for t in triples):
                    raise RuntimeError(f"Invalid spatial parser result: {triples}")
                results[text] = sorted(map(list, set(map(tuple, triples))))
                if connection:
                    connection.execute("INSERT OR REPLACE INTO parses VALUES (?, ?)", (key(text), json.dumps(results[text])))
            if connection:
                connection.commit()
        return [results[text] for text in requests]
    finally:
        if connection:
            connection.close()


def read_instances(source, image_root=None):
    """Read caption rows or flatten canonical evaluation answers in their original order."""
    rows, seen = [], set()
    image_root = Path(image_root) if image_root is not None else Path(source).resolve().parent
    with Path(source).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {number}: each JSONL row must be an object")
            if "caption" in row:
                if not isinstance(row["caption"], str) or not isinstance(row.get("label", ""), str):
                    raise ValueError(f"Line {number}: caption and label must be strings")
                rows.append(row)
                continue
            from .dataset import _dataset_answer, _image_value, parse_answer, image_path_and_id
            _, default_id = image_path_and_id(_image_value(row), image_root)
            image_id = row.get("image_id", default_id)
            if isinstance(image_id, bool) or not isinstance(image_id, (str, int)) or not str(image_id):
                raise ValueError(f"Line {number}: image_id must be a nonempty string or integer")
            if str(image_id) in seen:
                raise ValueError(f"Line {number}: duplicate image ID {image_id}")
            seen.add(str(image_id))
            objects, _, _ = parse_answer(_dataset_answer(row), strict=True)
            rows.extend({"image_id": image_id, "instance_id": f"g{i}",
                         "label": obj["label"], "caption": obj["caption"]}
                        for i, obj in enumerate(objects))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            temporary.chmod(path.stat().st_mode & 0o777)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Evaluation JSONL or instance rows with caption/label")
    parser.add_argument("output", type=Path, help="One relation GT row per instance")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path)
    parser.add_argument("--image-root", type=Path,
                        help="Evaluation image root; defaults to the input JSONL directory")
    parser.add_argument("--categories", type=Path, default=CATEGORY_PATH)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    if args.output.resolve() in {args.input.resolve(), args.categories.resolve(), OBJECT_PATH.resolve()}:
        parser.error("Output must not overwrite input or parser resources")
    rows = read_instances(args.input, args.image_root)
    triples = parse_captions([row["caption"] for row in rows], args.model_dir,
        labels=[row.get("label", "") for row in rows], categories_path=args.categories,
        library_dir=args.library_dir, batch_size=args.batch_size, cache_dir=args.cache_dir,
        threads=args.threads, timeout=args.timeout, progress=True)
    output = [dict({key: row[key] for key in ("image_id", "instance_id", "label", "caption") if key in row},
                   relations=relations, relation_parser=PARSER_VERSION)
              for row, relations in zip(rows, triples)]
    write_jsonl(args.output, output)
    print(f"Wrote {len(rows)} captions, {sum(len(r['relations']) for r in output)} spatial relations "
          f"to {args.output}")


if __name__ == "__main__":
    main()
