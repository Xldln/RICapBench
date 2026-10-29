#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
# Override SPICE_MODEL_DIR to store the resources elsewhere.
SPICE_MODEL_DIR="${SPICE_MODEL_DIR:-${SCRIPT_DIR}/third_party/stanford}"

command -v java >/dev/null || { echo "Install a JDK (Java 11 or later) first; see README.md." >&2; exit 1; }
command -v curl >/dev/null || { echo "curl is required to download the Java resources." >&2; exit 1; }
JAVA_MODULES="$(java --list-modules 2>/dev/null)" || { echo "Java 11 or later is required." >&2; exit 1; }
[[ "$JAVA_MODULES" =~ jdk.compiler@([0-9]+) ]] && (( BASH_REMATCH[1] >= 11 )) || { echo "A JDK with Java 11 or later and jdk.compiler is required." >&2; exit 1; }
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10 or later is required"'
"$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"

mkdir -p "$SPICE_MODEL_DIR"
URLS=(
  "https://repo.maven.apache.org/maven2/edu/stanford/nlp/stanford-corenlp/3.6.0/stanford-corenlp-3.6.0.jar"
  "https://repo.maven.apache.org/maven2/edu/stanford/nlp/stanford-corenlp/3.6.0/stanford-corenlp-3.6.0-models.jar"
  "https://repo.maven.apache.org/maven2/org/slf4j/slf4j-api/1.7.12/slf4j-api-1.7.12.jar"
  "https://repo.maven.apache.org/maven2/org/slf4j/slf4j-simple/1.7.21/slf4j-simple-1.7.21.jar"
)
for url in "${URLS[@]}"; do
  target="$SPICE_MODEL_DIR/${url##*/}"
  if [[ -s "$target" ]]; then
    echo "Using existing resource: $target"
    continue
  fi
  curl -fL --retry 3 "$url" -o "$target.part"
  mv "$target.part" "$target"
done
printf 'Environment resources are ready. Use --spice-model-dir "%s"\n' "$SPICE_MODEL_DIR"
