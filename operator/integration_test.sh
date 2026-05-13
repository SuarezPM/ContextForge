#!/usr/bin/env bash
# =============================================================================
# integration_test.sh — Kind-based CRD + manifest smoke test for Sprint 2
#
# WHAT THIS TESTS
#   This script validates the CRD schema and sample CR manifest flow by:
#     1. Applying the CRD to a real Kubernetes API server (inside kind).
#     2. Applying the sample ApohraContextForgeCluster CR.
#     3. Asserting the CR can be retrieved with kubectl.
#
# WHAT THIS DOES NOT TEST
#   The ApohraContextForgeCluster operator controller (Reconcile()) is NOT
#   deployed in Sprint 2.  The Go binary has not been built yet — that is a
#   Sprint 3 deliverable.  Consequently:
#     - Worker Deployments will NOT be created (no controller running).
#     - Redis sidecar will NOT be provisioned (no controller running).
#     - Status fields (readyWorkers, phase) will remain empty.
#
#   What this test DOES verify:
#     - The CRD YAML is accepted by the Kubernetes API server.
#     - The sample CR conforms to the CRD OpenAPI v3 schema.
#     - kubectl get/apply round-trips work correctly.
#
# PREREQUISITES
#   - kind  (https://kind.sigs.k8s.io) — skips gracefully if not installed.
#   - kubectl — required when kind is present.
#   - docker / podman — required by kind.
#
# USAGE
#   bash operator/integration_test.sh
#
# EXIT CODES
#   0 — SKIP (kind not installed) OR all assertions passed.
#   1 — assertion failed.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve paths relative to the repo root regardless of cwd.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CRD_MANIFEST="${REPO_ROOT}/operator/config/crd/bases/contextforge.apohara.dev_apoharacontextforgeclusters.yaml"
SAMPLE_CR="${REPO_ROOT}/operator/config/samples/contextforge_v1alpha1_apoharacontextforgecluster.yaml"
KIND_CLUSTER_NAME="apohara-contextforge-test"

# ---------------------------------------------------------------------------
# Guard: skip gracefully when kind is not installed.
# ---------------------------------------------------------------------------
if ! command -v kind &>/dev/null; then
    echo "SKIP: kind not installed — skipping integration test"
    echo "      Install kind from https://kind.sigs.k8s.io to run the full suite."
    exit 0
fi

if ! command -v kubectl &>/dev/null; then
    echo "SKIP: kubectl not installed — skipping integration test"
    echo "      kubectl is required alongside kind."
    exit 0
fi

echo "=== Apohara ContextForge — Sprint 2 Integration Test ==="
echo ""
echo "kind version : $(kind version)"
echo "kubectl      : $(kubectl version --client --short 2>/dev/null || kubectl version --client)"
echo ""

# ---------------------------------------------------------------------------
# Cleanup on exit (success or failure).
# ---------------------------------------------------------------------------
cleanup() {
    local exit_code=$?
    echo ""
    echo "--- cleanup: deleting kind cluster '${KIND_CLUSTER_NAME}' ---"
    kind delete cluster --name "${KIND_CLUSTER_NAME}" 2>/dev/null || true
    exit "${exit_code}"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 1. Create the kind cluster.
# ---------------------------------------------------------------------------
echo "--- Step 1: create kind cluster '${KIND_CLUSTER_NAME}' ---"
kind create cluster --name "${KIND_CLUSTER_NAME}" --wait 60s
echo ""

# Point kubectl at the new cluster.
kubectl cluster-info --context "kind-${KIND_CLUSTER_NAME}"
echo ""

# ---------------------------------------------------------------------------
# 2. Apply the CRD.
# ---------------------------------------------------------------------------
echo "--- Step 2: apply CRD ---"
echo "  file: ${CRD_MANIFEST}"
kubectl apply --context "kind-${KIND_CLUSTER_NAME}" -f "${CRD_MANIFEST}"
echo ""

# Wait for the CRD to become Established.
echo "--- Step 2a: wait for CRD to be Established ---"
kubectl wait \
    --context "kind-${KIND_CLUSTER_NAME}" \
    --for=condition=Established \
    --timeout=30s \
    crd/apoharacontextforgeclusters.contextforge.apohara.dev
echo ""

# ---------------------------------------------------------------------------
# 3. Apply the sample CR.
# ---------------------------------------------------------------------------
echo "--- Step 3: apply sample CR ---"
echo "  file: ${SAMPLE_CR}"
kubectl apply --context "kind-${KIND_CLUSTER_NAME}" -f "${SAMPLE_CR}"
echo ""

# ---------------------------------------------------------------------------
# 4. Wait briefly and then assert the CR can be retrieved.
#    (The operator is not deployed so no reconciliation happens; we only
#    verify the API server accepted the CR against the CRD schema.)
# ---------------------------------------------------------------------------
echo "--- Step 4: wait 5s then assert CR exists ---"
sleep 5

echo "  kubectl get apoharacontextforgecluster:"
kubectl get apoharacontextforgecluster \
    --context "kind-${KIND_CLUSTER_NAME}" \
    -n default

# Specific assertion — the sample CR named 'my-contextforge-cluster' must exist.
echo ""
echo "  asserting 'my-contextforge-cluster' exists..."
kubectl get apoharacontextforgecluster my-contextforge-cluster \
    --context "kind-${KIND_CLUSTER_NAME}" \
    -n default \
    -o jsonpath='{.spec.workerCount}' | grep -q "4" \
    && echo "  PASS: spec.workerCount=4 confirmed" \
    || { echo "  FAIL: spec.workerCount assertion failed"; exit 1; }

# Assert status subresource is accessible (even if empty — controller not running).
kubectl get apoharacontextforgecluster my-contextforge-cluster \
    --context "kind-${KIND_CLUSTER_NAME}" \
    -n default \
    -o jsonpath='{.status}' >/dev/null \
    && echo "  PASS: status subresource accessible" \
    || { echo "  FAIL: status subresource inaccessible"; exit 1; }

echo ""
echo "=== All assertions passed ==="
echo ""
echo "NOTE: The Reconcile() controller was NOT deployed in this test."
echo "      Worker Deployments and the Redis sidecar are Sprint 3+ deliverables."
echo "      This test validates CRD schema acceptance and CR round-trip only."
echo ""
# Cleanup runs via trap EXIT.
