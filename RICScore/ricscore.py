"""RICScore: see README.md for the scoring protocol and configuration."""

from __future__ import annotations

import argparse
from typing import Sequence
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from shapely.geometry import Polygon

import networkx as nx
import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

# Resolve an explicitly relocated parser before importing it or its rule tables.
if __name__ == "__main__":
    bootstrap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    bootstrap.add_argument("--spatial-parser-dir", type=Path)
    options, _ = bootstrap.parse_known_args()
    if options.spatial_parser_dir is not None:
        directory = options.spatial_parser_dir.expanduser().resolve()
        required = ("__init__.py", "parser.py", "SpatialRelationParser.java",
                    "spatial_predicates.tsv", "fact_equivalence.tsv",
                    "relation_facts.py", "RICRelationScorer.java", "object_landmarks.txt")
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            bootstrap.error(f"Invalid --spatial-parser-dir {directory}: missing {', '.join(missing)}")
        spec = importlib.util.spec_from_file_location(
            "spatial_relations", directory / "__init__.py",
            submodule_search_locations=[str(directory)])
        module = importlib.util.module_from_spec(spec)
        sys.modules["spatial_relations"] = module
        spec.loader.exec_module(module)

from spatial_relations.parser import CATEGORY_PATH, OBJECT_PATH, object_names, category_names, resolve_subject, normalize_predicate
from spatial_relations import PARSER_VERSION, SPATIAL_PREDICATES, parse_captions
from spatial_relations.relation_facts import RULES_PATH, normalize_relation_facts, score_relation_facts

DEFAULT_GT_RELATIONS = Path(__file__).resolve().parent / "data" / "RICapBench" / "evaluation-spatial-gt-relations.jsonl"

def _polygon(box: Sequence[float]) -> list[tuple[float, float]]:
    cx, cy, width, height, angle = box
    cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    corners = []
    for x, y in ((-width / 2, -height / 2), (width / 2, -height / 2),
                 (width / 2, height / 2), (-width / 2, height / 2)):
        corners.append((cx + x * cosine - y * sine, cy + x * sine + y * cosine))
    return corners


def rotated_iou(first: Sequence[float], second: Sequence[float]) -> float:
    first_polygon, second_polygon = Polygon(_polygon(first)), Polygon(_polygon(second))
    intersection = first_polygon.intersection(second_polygon).area
    union = first_polygon.area + second_polygon.area - intersection
    return intersection / union if union > 0 else 0.0




def _text(value):
    if not isinstance(value, str) or "|||" in value:
        raise ValueError("Captions must be strings without the METEOR delimiter '|||'")
    return " ".join(value.split())


def _identifier(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value):
        raise ValueError("Image and instance IDs must be non-empty strings or integers")
    return str(value)


def validate_records(document):
    """Normalize IDs/text without dropping malformed predictions or unlabeled GTs."""
    if not isinstance(document, dict) or not isinstance(document.get("images"), list):
        raise ValueError("Input must be an object containing an 'images' list")
    records, image_ids = [], set()
    for record in document["images"]:
        image_id = _identifier(record["image_id"])
        if image_id in image_ids:
            raise ValueError(f"Duplicate image_id: {image_id}")
        image_ids.add(image_id)
        prediction_format_valid = record.get("prediction_format_valid", True)
        if type(prediction_format_valid) is not bool:
            raise ValueError(f"{image_id}: prediction_format_valid must be boolean")
        invalid_count = record.get("invalid_prediction_count", 0)
        if type(invalid_count) is not int or invalid_count < 0:
            raise ValueError(f"{image_id}: invalid_prediction_count must be a non-negative integer")
        if prediction_format_valid and invalid_count:
            raise ValueError(f"{image_id}: canonical predictions cannot contain invalid objects")
        normalized = {"image_id": image_id, "prediction_format_valid": prediction_format_valid,
                      "invalid_prediction_count": invalid_count}
        for side in ("ground_truth", "predictions"):
            if not isinstance(record.get(side), list):
                raise ValueError(f"{image_id}: {side} must be a list, possibly empty")
            objects, ids = [], set()
            for raw in record[side]:
                item = dict(raw)
                item["id"] = _identifier(item["id"])
                if item["id"] in ids:
                    raise ValueError(f"{image_id}/{side}: duplicate instance ID {item['id']}")
                ids.add(item["id"])
                item.setdefault("label", "")
                if not isinstance(item["label"], str) or (side == "ground_truth" and not item["label"].strip()):
                    raise ValueError(f"{image_id}/{side}: label must be a string; GT requires a non-empty label")
                if side == "ground_truth":
                    refs = item.get("captions", [])
                    if not isinstance(refs, list):
                        raise ValueError("GT captions must be a list of reference strings")
                    item["captions"] = list(dict.fromkeys(filter(None, map(_text, refs))))
                else:
                    item["caption"] = _text(item.get("caption", ""))
                objects.append(item)
            normalized[side] = sorted(objects, key=lambda item: item["id"])
        records.append(normalized)
    return sorted(records, key=lambda record: record["image_id"])


def _mask(value, base_dir):
    if isinstance(value, str):
        with Image.open(base_dir / value) as image:
            array = np.array(image)
    elif isinstance(value, dict):
        from pycocotools import mask as mask_utils

        size, counts = value.get("size"), value.get("counts")
        if (not isinstance(size, list) or len(size) != 2
                or any(type(n) is not int or n <= 0 for n in size)):
            raise ValueError("COCO RLE size must be [positive height, positive width]")
        if isinstance(counts, list):
            if any(type(n) is not int or n < 0 for n in counts) or sum(counts) != math.prod(size):
                raise ValueError("Uncompressed RLE counts must sum to height * width")
            rle = mask_utils.frPyObjects(value, *size)
        elif isinstance(counts, str):
            rle = {"size": size, "counts": counts.encode("ascii")}
        else:
            raise ValueError("COCO RLE counts must be an integer list or compressed string")
        array = mask_utils.decode(rle)
    elif isinstance(value, list):
        array = np.asarray(value)
    else:
        raise ValueError("mask must be a binary PNG path, 2D binary array, or COCO RLE")
    if array.ndim != 2 or not array.size or not np.isin(array, (0, 1, 255)).all():
        raise ValueError("Masks must be non-empty 2D arrays containing only 0/1 or 0/255")
    return array != 0


def _rbox(value):
    if (not isinstance(value, list) or len(value) != 5
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
            or value[2] <= 0 or value[3] <= 0):
        raise ValueError("rbox must be finite [cx, cy, width>0, height>0, angle_degrees]")
    return value


def optimal_matching(ious, allowed, caption_scores=None):
    """Exactly maximize (cardinality, total IoU, total caption accuracy), in that order."""
    graph = nx.Graph()
    edges = []
    for p, g in zip(*np.nonzero(allowed)):
        caption = float(caption_scores[p, g]) if caption_scores is not None else 0.0
        if not math.isfinite(caption) or caption < 0:
            raise ValueError("Caption matching scores must be finite and non-negative")
        edges.append((p, g, float(ious[p, g]).as_integer_ratio(), caption.as_integer_ratio()))
    # Float denominators are powers of two: integer weights preserve their exact order.
    denominator = max((iou[1] for _, _, iou, _ in edges), default=1)
    caption_denominator = max((caption[1] for _, _, _, caption in edges), default=1)
    caption_weights = [num * (caption_denominator // den) for _, _, _, (num, den) in edges]
    # One integer unit of IoU dominates every possible change in the caption sum.
    iou_multiplier = sum(caption_weights) + 1
    for (p, g, (numerator, divisor), _), caption_weight in zip(edges, caption_weights):
        graph.add_edge(("p", int(p)), ("g", int(g)),
                       weight=numerator * (denominator // divisor) * iou_multiplier + caption_weight)
    pairs = []
    for left, right in nx.max_weight_matching(graph, maxcardinality=True):
        p, g = (left, right) if left[0] == "p" else (right, left)
        pairs.append((p[1], g[1]))
    return sorted(pairs)


def match_records(records, geometry, base_dir, *, caption_scorer=None, iou_threshold=0.5):
    if not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be finite and in (0, 1]")
    matches, tp, fp, fn = [], 0, 0, 0
    jobs, candidates = [], []
    for record in records:
        # Content order makes the last, fully tied decision independent of generated IDs/order.
        def order(obj):
            text = tuple(sorted(obj["captions"])) if "captions" in obj else (obj["caption"],)
            return json.dumps(obj[geometry], sort_keys=True), text, obj["id"]
        predictions = sorted(record["predictions"], key=order)
        truths = sorted(record["ground_truth"], key=order)
        convert = (lambda obj: _mask(obj["mask"], base_dir)) if geometry == "mask" else (
            lambda obj: _rbox(obj["rbox"]))
        pred_shapes, gt_shapes = list(map(convert, predictions)), list(map(convert, truths))
        if geometry == "mask":
            shapes = {value.shape for value in pred_shapes + gt_shapes}
            if len(shapes) > 1:
                raise ValueError(f"{record['image_id']}: masks must use the same pixel grid")
        ious = np.zeros((len(predictions), len(truths)))
        allowed = np.zeros_like(ious, dtype=bool)
        for p, prediction in enumerate(predictions):
            for g, truth in enumerate(truths):
                if geometry == "mask":
                    union = np.count_nonzero(pred_shapes[p] | gt_shapes[g])
                    iou = np.count_nonzero(pred_shapes[p] & gt_shapes[g]) / union if union else 0.0
                else:
                    iou = rotated_iou(pred_shapes[p], gt_shapes[g])
                ious[p, g], allowed[p, g] = iou, iou >= iou_threshold
        jobs.append((record, predictions, truths, ious, allowed))
        for p, g in zip(*np.nonzero(allowed)):
            if truths[g]["captions"]:
                candidates.append({"image_id": record["image_id"], "prediction_id": predictions[p]["id"],
                                   "ground_truth_id": truths[g]["id"], "caption": predictions[p]["caption"],
                                   "references": truths[g]["captions"]})
    # ponytail: score all eligible edges; prune dominated edges if caption scoring becomes a bottleneck.
    scores = caption_scorer(candidates) if caption_scorer and candidates else {}
    for record, predictions, truths, ious, allowed in jobs:
        caption_weights = np.zeros_like(ious)
        for p, g in zip(*np.nonzero(allowed)):
            key = (record["image_id"], predictions[p]["id"], truths[g]["id"])
            if key in scores:
                caption_weights[p, g] = _caption_accuracy(scores[key])
        pairs = optimal_matching(ious, allowed, caption_weights)
        tp += len(pairs)
        fp += len(predictions) - len(pairs) + record.get("invalid_prediction_count", 0)
        fn += len(truths) - len(pairs)
        for p, g in pairs:
            key = (record["image_id"], predictions[p]["id"], truths[g]["id"])
            matches.append({"image_id": record["image_id"], "prediction_id": predictions[p]["id"],
                            "ground_truth_id": truths[g]["id"], "iou": float(ious[p, g]),
                            "prediction_label": predictions[p]["label"], "ground_truth_label": truths[g]["label"],
                            "caption": predictions[p]["caption"], "references": truths[g]["captions"],
                            "caption_scores": scores.get(key)})
    return matches, tp, fp, fn


def _java(arguments, *, input_text=None, timeout=600):
    result = subprocess.run(["java", *arguments], input=input_text, text=True,
                            capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Java scorer failed ({result.returncode}): {result.stderr[-4000:]}")
    return result.stdout


def tokenize(captions):
    from pycocoevalcap.tokenizer import ptbtokenizer

    jar = Path(ptbtokenizer.__file__).with_name(ptbtokenizer.STANFORD_CORENLP_3_4_1_JAR)
    if not captions:
        return []
    output = _java(["-cp", str(jar), "edu.stanford.nlp.process.PTBTokenizer",
                    "-preserveLines", "-lowerCase"], input_text="\n".join(captions) + "\n")
    lines = output.splitlines()
    if len(lines) != len(captions):
        raise RuntimeError("PTB tokenizer returned an unexpected number of lines")
    return [" ".join(word for word in line.split() if word not in ptbtokenizer.PUNCTUATIONS)
            for line in lines]


def fixed_cider(references, predictions):
    """Compute IDF on every GT document, then return scores only for supplied predictions."""
    from pycocoevalcap.cider.cider_scorer import CiderScorer

    if not references:
        return {}
    if not any(ref.strip() for refs in references.values() for ref in refs):
        return dict.fromkeys(predictions, 0.0)
    scorer = CiderScorer(n=4, sigma=6.0)
    keys = sorted(references)
    for key in keys:
        scorer += (predictions.get(key, ""), references[key])
    # All GT instances remain in the corpus even when the model misses them.
    _, scores = scorer.compute_score()
    return {key: float(score) for key, score in zip(keys, scores) if key in predictions}


def meteor_scores(references, predictions):
    """Return per-pair scores from the standard METEOR scorer's default configuration."""
    if not predictions:
        return {}
    from pycocoevalcap.meteor.meteor import Meteor

    keys = sorted(predictions)
    scorer = Meteor()
    try:
        _, values = scorer.compute_score(
            {key: references[key] for key in keys},
            {key: [predictions[key]] for key in keys})
        return {key: float(value) for key, value in zip(keys, values)}
    finally:
        scorer.meteor_p.kill()
        scorer.meteor_p.wait()
        # The upstream wrapper leaves its lock held if scoring raises.
        if scorer.lock.locked():
            scorer.lock.release()


def _batches(values, size):
    if size <= 0:
        raise ValueError("SPICE batch size must be positive")
    for start in range(0, len(values), size):
        yield values[start:start + size]


def spice_scores(references, predictions, model_dir=None, *, batch_size=1000,
                 cache_dir=None, timeout=3600):
    from pycocoevalcap.spice import spice

    package_dir = Path(spice.__file__).parent
    lib = Path(model_dir) if model_dir else package_dir / "lib"
    required = [lib / f"stanford-corenlp-3.6.0{suffix}.jar" for suffix in ("", "-models")]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(
            "SPICE requires stanford-corenlp-3.6.0.jar and stanford-corenlp-3.6.0-models.jar; "
            "provide their directory with --spice-model-dir. No approximate score is substituted.")
    classpath = os.pathsep.join([str(package_dir / "spice-1.0.jar"),
                                str(package_dir / "lib" / "*"), str(lib.resolve() / "*")])
    keys = sorted(predictions)
    batches = list(_batches(keys, batch_size))
    cache_dir = Path(cache_dir or Path.home() / ".cache" / "ricscore" / "spice")
    cache_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for number, batch in enumerate(batches, 1):
        print(f"SPICE batch {number}/{len(batches)} ({len(batch)} descriptions)",
              file=sys.stderr, flush=True)
        with tempfile.TemporaryDirectory(prefix="ricscore-spice-") as temporary:
            source, target = Path(temporary) / "input.json", Path(temporary) / "output.json"
            source.write_text(json.dumps([
                {"image_id": i, "test": predictions[key], "refs": references[key]}
                for i, key in enumerate(batch)], ensure_ascii=False), encoding="utf-8")
            _java(["-Xmx8G", "-cp", classpath, str(Path(__file__).with_name("SpiceOAScorer.java")),
                   str(source), str(target), str(cache_dir)],
                  timeout=timeout)
            rows = json.loads(target.read_text(encoding="utf-8"))
        if len(rows) != len(batch) or {row["image_id"] for row in rows} != set(range(len(batch))):
            raise RuntimeError(f"SPICE batch {number} returned missing or duplicate results")
        result.update({batch[row["image_id"]]: row["scores"] for row in rows})
    return result


def _counts(scores, subset):
    result = tuple(scores[subset][key] for key in ("tp", "fp", "fn"))
    if any(type(value) not in (int, float) or not math.isfinite(value)
           or value < 0 or int(value) != value for value in result):
        raise RuntimeError(f"Invalid SPICE {subset} counts: {result}")
    return tuple(map(int, result))


def _spice_accuracy(scores, subsets=("Object", "Attribute")):
    """Pool disjoint fact counts within a caption pair before computing F1."""
    if subsets == ("All",) and "f" in scores["All"]:
        # Official many-to-one counts can be negative; preserve its reported auxiliary F1.
        value = scores["All"]["f"]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise RuntimeError(f"Invalid official SPICE All score: {value}")
        return value
    counts = [_counts(scores, subset) for subset in subsets]
    tp, fp, fn = (sum(values) for values in zip(*counts))
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def _caption_accuracy(row):
    spice = _spice_accuracy(row["spice"])
    return (row["meteor"] + row["cider_d"] / 10 + spice) / 3


def normalize_gt_triples(triples):
    """Normalize predicate aliases and grouped endpoints before fact scoring."""
    import re
    names = object_names()
    output = []
    for subject, predicate, obj in triples:
        subject, predicate, obj = (" ".join(value.split()) for value in (subject, predicate, obj))
        predicate = normalize_predicate(predicate)
        members = []
        for member in re.split(r"\s*&\s*|\s+and\s+", obj):
            member = re.sub(r"^(?:a|an|the)\s+", "", member)
            if member not in names:
                member = re.sub(r"^marked\s+", "", member)
            members.append(member)
        if predicate == 'between':
            output.append([subject, predicate, ' & '.join(sorted(members))])
        else:
            output.extend([subject, predicate, member] for member in members)
    return output


def load_gt_relations(source, records, *, model_dir=None, timeout=3600):
    """Require the fixed relation file to cover exactly the input GT instances."""
    expected = {(record["image_id"], obj["id"]): obj
                for record in records for obj in record["ground_truth"]}
    relations = {}
    categories = category_names()
    landmarks = object_names()
    with Path(source).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (_identifier(row["image_id"]), _identifier(row["instance_id"]))
            if key in relations or key not in expected:
                raise ValueError(f"{source}:{number}: duplicate or unexpected relation instance {key}")
            gt = expected[key]
            caption = _text(row["caption"])
            if row["label"] != gt["label"] or ([caption] if caption else []) != gt["captions"]:
                raise ValueError(f"{source}:{number}: relation GT label/caption disagrees with evaluation GT {key}")
            triples = row["relations"]
            if not isinstance(triples, list) or any(
                    not isinstance(triple, list) or len(triple) != 3
                    or any(not isinstance(word, str) or not word.strip() for word in triple)
                    for triple in triples):
                raise ValueError(f"{source}:{number}: relations must be [subject, predicate, object] triples")
            triples = normalize_gt_triples(triples)
            if row.get("relation_parser") != PARSER_VERSION:
                raise ValueError(f"{source}:{number}: incompatible spatial parser version")
            if any(triple[1] not in SPATIAL_PREDICATES for triple in triples):
                raise ValueError(f"{source}:{number}: GT contains noncanonical spatial predicates; regenerate spatial GT")
            if any(triple[0] not in categories or any(member not in landmarks
                   for member in triple[2].split(" & ")) for triple in triples):
                raise ValueError(f"{source}:{number}: forbidden subject/object; regenerate spatial GT")
            subject = resolve_subject(caption, gt["label"], categories=categories)
            if any(not subject or triple[0] != subject["subject"] for triple in triples):
                raise ValueError(f"{source}:{number}: relation subject disagrees with caption/category")
            relations[key] = sorted(set(tuple(_text(word) for word in triple) for triple in triples))
    missing = expected.keys() - relations.keys()
    if missing:
        raise ValueError(f"Relation GT missing {len(missing)} instances, including {sorted(missing)[:3]}")
    keys = list(relations)
    normalized = normalize_relation_facts([list(map(list, relations[key])) for key in keys], model_dir=model_dir, timeout=timeout)
    return {key: list(map(tuple, row)) for key, row in zip(keys, normalized)}


def relation_scores(jobs, model_dir=None, *, batch_size=1000, cache_dir=None, timeout=3600):
    """Use the independent spatial parser, then compare with fixed spatial GT triples."""
    if not jobs:
        return []
    from pycocoevalcap.spice import spice

    package_dir = Path(spice.__file__).parent
    lib = Path(model_dir) if model_dir else package_dir / "lib"
    if not all((lib / f"stanford-corenlp-3.6.0{suffix}.jar").is_file() for suffix in ("", "-models")):
        raise FileNotFoundError("Relation scoring requires Stanford CoreNLP 3.6.0 jars; provide --spice-model-dir")
    cache_dir = Path(cache_dir or Path.home() / ".cache" / "ricscore" / "spice")
    cache_dir.mkdir(parents=True, exist_ok=True)
    extracted = parse_captions([job["caption"] for job in jobs], lib,
        labels=[job.get("label", "") for job in jobs],
        library_dir=None if any(lib.glob("slf4j-api*.jar")) else package_dir / "lib", batch_size=batch_size,
        cache_dir=cache_dir / "spatial", timeout=timeout, progress=True)
    jobs = [dict(job, predicted_relations=triples) for job, triples in zip(jobs, extracted)]
    rows = score_relation_facts(jobs, batch_size=batch_size, timeout=timeout, model_dir=lib)
    return [_counts({"Relation": row}, "Relation") for row in rows]


def score_all_relations(records, matches, ground_truth, scorer, *, include_matched=False, image_counts=None):
    """Score once; optionally return (all counts, matched-instance counts)."""
    paired = {(match["image_id"], match["prediction_id"]): match["ground_truth_id"]
              for match in matches}
    matched_gt = {(match["image_id"], match["ground_truth_id"]) for match in matches}
    totals = [0, 0, 0]
    matched_totals = [0, 0, 0]
    if image_counts is not None:
        image_counts.update({r['image_id']: [[0, 0, 0], [0, 0, 0]] for r in records})

    def accumulate(image_id, counts, matched):
        for i, count in enumerate(counts):
            totals[i] += count
            if matched:
                matched_totals[i] += count
            if image_counts is not None:
                image_counts[image_id][0][i] += count
                if matched:
                    image_counts[image_id][1][i] += count

    for key, triples in ground_truth.items():
        if key not in matched_gt:
            accumulate(key[0], (0, 0, len(triples)), False)
    jobs = []
    for record in records:
        for prediction in record["predictions"]:
            key = (record["image_id"], prediction["id"])
            gt_id = paired.get(key)
            triples = ground_truth[(key[0], gt_id)] if gt_id is not None else []
            if prediction["caption"]:
                jobs.append({"image_id": key[0], "caption": prediction["caption"], "label": prediction.get("label", ""), "relations": triples,
                             "matched": gt_id is not None})
            else:
                accumulate(key[0], (0, 0, len(triples)), gt_id is not None)
    rows = scorer(jobs) if jobs else []
    if len(rows) != len(jobs):
        raise RuntimeError("Relation scorer returned missing results")
    for job, counts in zip(jobs, rows):
        rtp, rfp, rfn = _counts({"Relation": dict(zip(("tp", "fp", "fn"), counts))}, "Relation")
        if rtp + rfn != len(job["relations"]):
            raise RuntimeError("Relation scorer changed the fixed GT relation count")
        accumulate(job["image_id"], (rtp, rfp, rfn), job["matched"])
    return (tuple(totals), tuple(matched_totals)) if include_matched else tuple(totals)


def summarize(tp, fp, fn, caption_rows, relations, matched_relations):
    """Combine conditional caption/relation scores; report full-scope relations separately."""
    denominator = tp + fp + fn
    result = {"TP": tp, "FP": fp, "FN": fn, "caption_pairs": len(caption_rows),
              "DetA": tp / denominator if denominator else None,
              "METEOR": None, "CIDEr_D": None, "CIDEr_D_normalized": None,
              "SPICE": None, "SPICE_OA": None, "SPICE_All": None,
              "CapA": None, "RelCap_F1": None,
              "RelCap_F1_matched": None, "RelCap_F1_all": None, "RICScore": None,
              "relation_TP": matched_relations[0], "relation_FP": matched_relations[1],
              "relation_FN": matched_relations[2],
              "relation_all_TP": relations[0], "relation_all_FP": relations[1],
              "relation_all_FN": relations[2], "reason": None}
    rtp, rfp, rfn = matched_relations
    if sum(matched_relations):
        result["RelCap_F1"] = 2 * rtp / (2 * rtp + rfp + rfn)
        result["RelCap_F1_matched"] = result["RelCap_F1"]
    if sum(relations):
        atp, afp, afn = relations
        result["RelCap_F1_all"] = 2 * atp / (2 * atp + afp + afn)
    if not denominator:
        result["reason"] = "empty_ground_truth_and_predictions"
    elif not tp:
        result.update(RICScore=0.0, reason="no_detection_matches")
    elif not caption_rows:
        result["reason"] = "no_caption_annotated_matches"
    else:
        spice_values = [_spice_accuracy(row["spice"]) for row in caption_rows]
        meteor = math.fsum(row["meteor"] for row in caption_rows) / len(caption_rows)
        cider = math.fsum(row["cider_d"] for row in caption_rows) / len(caption_rows)
        spice = math.fsum(spice_values) / len(spice_values)
        spice_all = math.fsum(_spice_accuracy(row["spice"], ("All",))
                             for row in caption_rows) / len(caption_rows)
        capa = math.fsum(_caption_accuracy(row) for row in caption_rows) / len(caption_rows)
        result.update(METEOR=meteor, CIDEr_D=cider, CIDEr_D_normalized=cider / 10,
                      SPICE=spice, SPICE_OA=spice, SPICE_All=spice_all,
                      CapA=capa, relation_TP=rtp, relation_FP=rfp, relation_FN=rfn)
        if not sum(matched_relations):
            result["reason"] = "no_matched_predicted_or_reference_relations"
        else:
            result["RICScore"] = (result["DetA"] * capa * result["RelCap_F1"]) ** (1 / 3)
    return result


def summarize_subsets(records, matches, image_counts):
    """Aggregate existing scores by source; retain the full evaluation CIDEr IDF corpus."""
    subsets = {}
    for source in ('CODrone', 'DIOR_R', 'FAIR1M'):
        members = [r for r in records if source in Path(r['image_id']).parts]
        if not members:
            continue
        ids = {r['image_id'] for r in members}
        paired = [m for m in matches if m['image_id'] in ids]
        tp = len(paired)
        fp = sum(len(r['predictions']) + r.get('invalid_prediction_count', 0) for r in members) - tp
        fn = sum(len(r['ground_truth']) for r in members) - tp
        relations, matched_relations = (
            tuple(sum(image_counts[i][scope][k] for i in ids) for k in range(3))
            for scope in (0, 1))
        subsets[source] = summarize(tp, fp, fn,
            [m['caption_scores'] for m in paired if m['references']], relations, matched_relations)
        subsets[source]['image_count'] = len(members)
    return subsets


def evaluate(document, *, geometry="rbox", base_dir=Path("."), spice_model_dir=None,
             spice_batch_size=1000, spice_cache_dir=None, spice_timeout=3600, iou_threshold=0.5,
             gt_relations=DEFAULT_GT_RELATIONS):
    if geometry not in ("mask", "rbox", "both"):
        raise ValueError("geometry must be mask, rbox or both")
    if spice_batch_size <= 0 or spice_timeout <= 0:
        raise ValueError("SPICE batch size and timeout must be positive")
    oa_scorer = Path(__file__).with_name("SpiceOAScorer.java")
    if not oa_scorer.is_file():
        raise FileNotFoundError(f"Missing CapA helper: {oa_scorer}; keep it beside ricscore.py")
    records = validate_records(document)
    relation_gt = load_gt_relations(gt_relations, records, model_dir=spice_model_dir, timeout=spice_timeout)
    modes = ("mask", "rbox") if geometry == "both" else (geometry,)
    references = {(record["image_id"], obj["id"]): obj["captions"]
                  for record in records for obj in record["ground_truth"] if obj["captions"]}
    digest = hashlib.sha256(json.dumps(
        [(key, refs) for key, refs in sorted(references.items())], ensure_ascii=False).encode()).hexdigest()
    relation_digest = hashlib.sha256(json.dumps(
        sorted(relation_gt.items()), ensure_ascii=False).encode()).hexdigest()
    report = {"protocol": "RICScore-2026-10-17-spatial-v0.1.0-category-grounded-spice-oa-one-to-one-matched-relations-class-agnostic-optional-ref-default-meteor", "primary_geometry": modes[0],
              "relation_gt_sha256": relation_digest,
              "relation_gt_instances": len(relation_gt),
              "relation_gt_count": sum(map(len, relation_gt.values())),
              "relation_scope": "matched_instances",
              "subset_scoring": "aggregate_global_matches_and_scores_with_full_evaluation_IDF",
              "relation_all_scope": "all_GT_instances_and_all_valid_predictions",
              "ricscore_components": ["DetA", "CapA", "RelCap_F1_matched"],
              "capa_components": ["METEOR", "CIDEr_D/10", "SPICE_OA"],
              "spice_oa_subsets": ["Object", "Attribute"],
              "spice_oa_aggregation": "pool_counts_per_pair_then_mean_pair_F1",
              "spice_oa_matching": "official_SPICE_semantic_equivalence_maximum_one_to_one",
              "spice_oa_scorer_sha256": hashlib.sha256(oa_scorer.read_bytes()).hexdigest(),
              "relation_matching": "equivalent_spatial_facts_SPICE_entities_deduplicated_one_to_one",
              "relation_parser": PARSER_VERSION,
              "relation_categories_sha256": hashlib.sha256(CATEGORY_PATH.read_bytes()).hexdigest(),
              "relation_objects_sha256": hashlib.sha256(OBJECT_PATH.read_bytes()).hexdigest(),
              "relation_fact_rules_sha256": hashlib.sha256(RULES_PATH.read_bytes()).hexdigest(),
              "category_mode": "class_agnostic",
              "matching_priority": ["cardinality", "total_iou", "total_CapA"],
              "invalid_prediction_count": sum(record["invalid_prediction_count"] for record in records),
              "invalid_prediction_policy": "invalid_rbox_objects_count_as_FP",
              "iou_threshold": iou_threshold, "idf_documents": len(references), "idf_corpus_sha256": digest,
              "prediction_format_valid_rate": (
                  sum(record["prediction_format_valid"] for record in records) / len(records)
                  if records else None),
              "versions": {name: importlib.metadata.version(name)
                           for name in ("networkx", "numpy", "shapely", "pycocoevalcap")},
              "warnings": [], "metrics": {}}
    if len(references) == 1:
        report["warnings"].append("CIDEr-D has zero IDF with a one-document reference corpus")
    tokenized_refs, score_cache = None, {}

    def score_candidates(candidates):
        nonlocal tokenized_refs
        pending = {(item["image_id"], item["prediction_id"], item["ground_truth_id"]): item
                   for item in candidates
                   if (item["image_id"], item["prediction_id"], item["ground_truth_id"]) not in score_cache}
        if not pending:
            return score_cache
        if tokenized_refs is None:
            flat = tokenize([ref for refs in references.values() for ref in refs])
            tokenized_refs, cursor = {}, 0
            for key, refs in references.items():
                tokenized_refs[key] = flat[cursor:cursor + len(refs)]
                cursor += len(refs)
        predictions = {key: item["caption"] for key, item in pending.items()}
        pair_refs = {key: item["references"] for key, item in pending.items()}
        tokenized_preds = dict(zip(predictions, tokenize(list(predictions.values()))))
        pair_tokenized_refs = {key: tokenized_refs[key[0], key[2]] for key in pending}
        # Score at most one candidate per GT in each round, preserving the original IDF corpus.
        groups = {}
        for key in pending:
            groups.setdefault((key[0], key[2]), []).append(key)
        cider = {}
        for index in range(max(map(len, groups.values()))):
            batch = {gt: keys[index] for gt, keys in groups.items() if index < len(keys)}
            values = fixed_cider(tokenized_refs, {gt: tokenized_preds[key] for gt, key in batch.items()})
            cider.update({key: values[gt] for gt, key in batch.items()})
        meteor = meteor_scores(pair_tokenized_refs, tokenized_preds)
        spice = spice_scores(pair_refs, predictions, spice_model_dir,
                             batch_size=spice_batch_size, cache_dir=spice_cache_dir,
                             timeout=spice_timeout)
        score_cache.update({key: {"meteor": meteor[key], "cider_d": cider[key], "spice": spice[key]}
                            for key in pending})
        return score_cache

    for mode in modes:
        matches, tp, fp, fn = match_records(records, mode, Path(base_dir),
                                           caption_scorer=score_candidates, iou_threshold=iou_threshold)
        rows = [match["caption_scores"] for match in matches if match["references"]]
        image_counts = {}
        relations, matched_relations = score_all_relations(records, matches, relation_gt, lambda jobs: relation_scores(
            jobs, spice_model_dir, batch_size=spice_batch_size,
            cache_dir=spice_cache_dir, timeout=spice_timeout), include_matched=True, image_counts=image_counts)
        report["metrics"][mode] = summarize(tp, fp, fn, rows, relations, matched_relations)
        report["metrics"][mode]["matches"] = matches
        report["metrics"][mode]["subsets"] = summarize_subsets(records, matches, image_counts)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("input", type=Path, help="JSON containing images, ground_truth and predictions")
    parser.add_argument("--spatial-parser-dir", type=Path,
                        help="Directory containing the independent parser's __init__.py, Java source and rule tables")
    parser.add_argument("--geometry", choices=("mask", "rbox", "both"), default="rbox")
    parser.add_argument("--gt-relations", type=Path,
                        default=DEFAULT_GT_RELATIONS,
                        help="Complete instance relation GT JSONL, validated against evaluation GT")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="Detection IoU threshold in (0, 1], default: 0.5")
    parser.add_argument("--spice-model-dir", type=Path, help="Directory containing Stanford CoreNLP 3.6.0 jars")
    parser.add_argument("--spice-batch-size", type=int, default=1000,
                        help="Descriptions per SPICE process (default: 1000)")
    parser.add_argument("--spice-cache-dir", type=Path,
                        default=Path.home() / ".cache" / "ricscore" / "spice",
                        help="Persistent SPICE cache directory")
    parser.add_argument("--spice-timeout", type=int, default=3600,
                        help="Timeout in seconds for each SPICE batch (default: 3600)")
    parser.add_argument("--output", type=Path, help="JSON report (default: stdout)")
    args = parser.parse_args()
    if args.output and args.output.resolve() in {args.input.resolve(), args.gt_relations.resolve()}:
        parser.error("--output must not overwrite the input or relation GT")
    try:
        document = json.loads(args.input.read_text(encoding="utf-8"))
        report = evaluate(document, geometry=args.geometry, base_dir=args.input.resolve().parent,
                          gt_relations=args.gt_relations,
                          iou_threshold=args.iou_threshold,
                          spice_model_dir=args.spice_model_dir,
                          spice_batch_size=args.spice_batch_size,
                          spice_cache_dir=args.spice_cache_dir,
                          spice_timeout=args.spice_timeout)
        output = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                                 prefix=args.output.name + ".", suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(output)
                temporary.replace(args.output)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        else:
            print(output, end="")
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"RICScore evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
