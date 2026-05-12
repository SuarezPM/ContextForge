#!/usr/bin/env bash
# Sprint 1 YAML validation script.
# Parses all YAML files under operator/ and charts/ with Python's yaml.safe_load.
# Exits 0 if all are valid; exits 1 on first parse error.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPERATOR_DIR="$ROOT/operator"
CHARTS_DIR="$ROOT/charts"

PASS=0
FAIL=0
ERRORS=()

validate_yaml() {
    local file="$1"
    if python3 -c "
import sys, yaml
try:
    yaml.safe_load(open('$file'))
except yaml.YAMLError as e:
    print('YAML parse error in $file:', e, file=sys.stderr)
    sys.exit(1)
"; then
        echo "  OK  $file"
        PASS=$((PASS + 1))
    else
        echo "  FAIL  $file"
        FAIL=$((FAIL + 1))
        ERRORS+=("$file")
    fi
}

echo "=== Apohara ContextForge — Sprint 1 YAML validation ==="
echo ""

echo "--- operator/ YAML files ---"
while IFS= read -r -d '' f; do
    validate_yaml "$f"
done < <(find "$OPERATOR_DIR" -name "*.yaml" -o -name "*.yml" | sort | tr '\n' '\0')

echo ""
echo "--- charts/ YAML files ---"
while IFS= read -r -d '' f; do
    # Skip Helm template files (they contain {{ }} which are not valid YAML alone)
    if [[ "$f" == *"/templates/"* && "$f" != *"_helpers.tpl" ]]; then
        echo "  SKIP (Helm template) $f"
        continue
    fi
    validate_yaml "$f"
done < <(find "$CHARTS_DIR" -name "*.yaml" -o -name "*.yml" | sort | tr '\n' '\0')

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="

if [[ $FAIL -gt 0 ]]; then
    echo "FAILED files:"
    for f in "${ERRORS[@]}"; do
        echo "  - $f"
    done
    exit 1
fi

echo "All YAML files are valid."
exit 0
