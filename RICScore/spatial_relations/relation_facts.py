"""Shared Java fact equivalence, semantic deduplication, and relation matching."""
import json
import os
from pathlib import Path
import subprocess
import tempfile


RULES_PATH = Path(__file__).with_name("fact_equivalence.tsv")


def score_relation_facts(rows, *, deduplicate=False, batch_size=5000, timeout=3600, model_dir=None):
    if batch_size <= 0 or timeout <= 0:
        raise ValueError("batch_size and timeout must be positive")
    if not rows:
        return []
    from pycocoevalcap.spice import spice
    package = Path(spice.__file__).parent
    classpath = os.pathsep.join([str(package / "spice-1.0.jar"), str(package / "lib" / "*")])
    if model_dir is not None:
        classpath += os.pathsep + str(Path(model_dir).resolve() / "*")
    output = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        with tempfile.TemporaryDirectory(prefix="ric-facts-") as directory:
            source, target = Path(directory) / "input.json", Path(directory) / "output.json"
            source.write_text(json.dumps(batch, ensure_ascii=False), encoding="utf-8")
            command = ["java", "-Xmx4G", "-cp", classpath,
                       str(Path(__file__).with_name("RICRelationScorer.java")),
                       str(source), str(target), str(RULES_PATH)]
            if deduplicate:
                command.append("--deduplicate")
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
            if result.returncode:
                raise RuntimeError(f"Relation fact scorer failed ({result.returncode}): {result.stderr[-6000:]}")
            parsed = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(parsed, list) or len(parsed) != len(batch):
            raise RuntimeError("Relation fact scorer returned the wrong number of rows")
        output.extend(parsed)
    return output


def normalize_relation_facts(rows, *, batch_size=5000, timeout=3600, model_dir=None):
    """Deduplicate per-instance equivalent facts; retain an original ordered triple."""
    result = score_relation_facts(rows, deduplicate=True, batch_size=batch_size, timeout=timeout, model_dir=model_dir)
    for row in result:
        if not isinstance(row, list) or any(
                not isinstance(t, list) or len(t) != 3
                or any(not isinstance(w, str) or not w.strip() for w in t) for t in row):
            raise RuntimeError("Relation normalizer returned invalid triples")
    return [sorted(row) for row in result]
