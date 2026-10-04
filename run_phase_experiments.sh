#!/usr/bin/env bash
# =============================================================================
# ASEM two-phase benchmark runner
#   Phase A (ingest)   build + persist one bank per system/conversation
#   Phase B (retrieve) load those banks, answer QA, swap ONLY the backbone
# =============================================================================
#
# The two phases talk to each other purely through a bank tag:
#
#   data/benchmarks/ingested_banks/<tag>/<System>/<locomo_000N>/<bank>.sqlite
#
# so Phase B never rebuilds memory. A model-size sweep therefore measures the
# backbone and nothing else. Same tag => same experiment; new tag => new one.
#
# ---------------------------------------------------------------------------
# LOCAL (HuggingFace) vs API backbones
# ---------------------------------------------------------------------------
#   local : configs/models/qwen*_hf.yaml  -> torch runs on this machine
#           CPU-only here, so 4B is slow; 1.5B is the sane default.
#   api   : configs/models/qwen*_api.yaml -> OpenAI-compatible /v1 endpoint
#           (vLLM, LM Studio, DeepSeek, proxy) configured via .env.
#
#   The API config's `model:` field MUST equal the server's served model name.
#   If your endpoint serves one model (e.g. DeepSeek), use --config for the
#   retrieval phase instead of --models-group, otherwise the sweep will ask
#   the endpoint for model names it does not host.
#
# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
#   bash run_phase_experiments.sh <mode>
#
# Modes:
#   smoke           Tiny end-to-end check (local HF, 10 QA pairs, 1 system)
#   hf-ingest       Phase A with the local HF backbone
#   hf-retrieve     Phase B over the local HF group, reusing the HF banks
#   hf-all          Phase A + Phase B (local HF)
#   api-ingest      Phase A with the API backbone
#   api-retrieve    Phase B over the API group, reusing the API banks
#   api-all         Phase A + Phase B (API)
#   sweep           Ingest once (local HF) then retrieve over every model
#   inspect         Show the manifest + bank tree for a tag
#   rerun           Delete saved predictions for a tag (forces fresh retrieval)
#   clean           Delete bank / work / result artifacts for a tag
#   env             Verify the conda env and the model configs
#   help            This text
#
# Common overrides (env vars):
#   TAG              Experiment name; tags derive as <TAG>_hf / <TAG>_api  (exp1)
#   HF_TAG, API_TAG  Set the two tags explicitly instead of deriving them
#   SYSTEMS          Systems to run                     (FastASEM ASEMv2)
#   LIMIT            Max QA pairs in Phase B only; empty = all
#                    NOTE: ingest always covers every conversation — LIMIT does
#                    not shrink Phase A.
#   METRICS          Metric names                       (em rougeL)
#   PER_CATEGORY     1 to add --per-category            (1)
#   HF_INGEST_CONFIG / API_INGEST_CONFIG / API_RETRIEVE_CONFIG
#   HF_GROUP / API_GROUP / ALL_GROUP                   (hf_small/api_small/all)
#   BANK_ROOT / RESULTS_DIR / WORK_ROOT                (defaults below)
#   DRY_RUN          1 to print commands without running them
#   EXTRA_ARGS       Extra flags appended verbatim to every phase command
#
# Examples:
#   TAG=thesis bash run_phase_experiments.sh hf-all
#   TAG=thesis LIMIT=200 SYSTEMS="FastASEM ASEMv2 ASEM" \
#       bash run_phase_experiments.sh hf-retrieve
#   TAG=api4b API_RETRIEVE_CONFIG=configs/models/qwen3_4b_api.yaml \
#       bash run_phase_experiments.sh api-all
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PYTHONPATH="${PYTHONPATH:-.}"
# Windows + faiss/torch: without this, imports die with "OMP: Error #15".
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

# Credentials for the API configs (OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL).
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-memory-r1}"

INPUT="${INPUT:-datasets/locomo/locomo10.json}"
REGISTRY="${REGISTRY:-configs/models/registry.yaml}"

BANK_ROOT="${BANK_ROOT:-data/benchmarks/ingested_banks}"
RESULTS_DIR="${RESULTS_DIR:-data/benchmarks/results/phased}"
WORK_ROOT="${WORK_ROOT:-data/benchmarks/retrieval_work}"

HF_INGEST_CONFIG="${HF_INGEST_CONFIG:-configs/models/qwen2.5_1.5b_hf.yaml}"
API_INGEST_CONFIG="${API_INGEST_CONFIG:-configs/models/qwen3_4b_api.yaml}"
API_RETRIEVE_CONFIG="${API_RETRIEVE_CONFIG:-}"

TAG_BASE="${TAG:-exp1}"
HF_TAG="${HF_TAG:-${TAG_BASE}_hf}"
API_TAG="${API_TAG:-${TAG_BASE}_api}"
SMOKE_TAG="${SMOKE_TAG:-smoke_hf}"
SMOKE_GROUP="${SMOKE_GROUP:-hf_small}"

HF_GROUP="${HF_GROUP:-hf_small}"
API_GROUP="${API_GROUP:-api_small}"
ALL_GROUP="${ALL_GROUP:-all}"

SYSTEMS="${SYSTEMS:-FastASEM ASEMv2}"
METRICS="${METRICS:-em rougeL}"
PER_CATEGORY="${PER_CATEGORY:-1}"
LIMIT="${LIMIT:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"

RUNNER="scripts/run_phase_benchmark.py"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { printf '\n=== %s ===\n' "$*"; }
warn() { printf '[warn] %s\n' "$*" >&2; }

run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if [ "${DRY_RUN}" = "1" ]; then
        return 0
    fi
    "$@"
}

ensure_env() {
    if [ "${DRY_RUN}" = "1" ]; then return 0; fi

    if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV}" = "${CONDA_ENV}" ]; then
        echo "[env] already in conda env '${CONDA_ENV}'"
        return 0
    fi

    local conda_sh="${CONDA_SH:-}"
    if [ -z "${conda_sh}" ]; then
        local cand
        for cand in \
            "${HOME}/anaconda3/etc/profile.d/conda.sh" \
            "${HOME}/miniconda3/etc/profile.d/conda.sh" \
            "${HOME}/miniforge3/etc/profile.d/conda.sh" \
            "/opt/conda/etc/profile.d/conda.sh" \
            "/home/datpd/miniconda3/etc/profile.d/conda.sh"
        do
            if [ -f "${cand}" ]; then conda_sh="${cand}"; break; fi
        done
    fi

    if [ -n "${conda_sh}" ] && [ -f "${conda_sh}" ]; then
        # shellcheck disable=SC1090
        . "${conda_sh}"
        # shellcheck disable=SC1091
        conda activate "${CONDA_ENV}"
        echo "[env] activated conda env '${CONDA_ENV}'"
    else
        warn "conda.sh not found; assuming 'python' already resolves to '${CONDA_ENV}'"
    fi
}

verify_env() {
    if [ ! -f "${INPUT}" ]; then
        echo "[env] dataset not found: ${INPUT}" >&2
        return 1
    fi
    if [ ! -f "${REGISTRY}" ]; then
        echo "[env] registry not found: ${REGISTRY}" >&2
        return 1
    fi
    if [ "${DRY_RUN}" = "1" ]; then return 0; fi
    python - <<'PY'
import sys
try:
    import yaml  # noqa: F401
    import asem  # noqa: F401
except Exception as exc:  # pragma: no cover - environment guard
    sys.exit(f"[env] import check failed: {exc}\n"
             f"      activate the project env (conda activate memory-r1) first")
print("[env] ok — yaml + asem import cleanly")
PY
}

# Phase A — build and persist the banks for one tag.
phase_ingest() {
    local tag="$1" config="$2"
    run python "${RUNNER}" \
        --phase ingest \
        --tag "${tag}" \
        --ingest-config "${config}" \
        --registry "${REGISTRY}" \
        --input "${INPUT}" \
        --bank-root "${BANK_ROOT}" \
        --results-dir "${RESULTS_DIR}" \
        --systems ${SYSTEMS} \
        ${EXTRA_ARGS}
}

# Phase B — reuse an existing bank tree; only the backbone changes.
# $1 = tag, $2 = "" (use group) or a config path for a single-model run.
phase_retrieve() {
    local tag="$1" single_config="${2:-}" group="${3:-}"

    local scope
    if [ -n "${single_config}" ]; then
        scope=(--config "${single_config}")
    else
        scope=(--models-group "${group}")
    fi

    run python "${RUNNER}" \
        --phase retrieve \
        --tag "${tag}" \
        "${scope[@]}" \
        --registry "${REGISTRY}" \
        --input "${INPUT}" \
        --bank-root "${BANK_ROOT}" \
        --results-dir "${RESULTS_DIR}" \
        --work-root "${WORK_ROOT}" \
        --systems ${SYSTEMS} \
        --metrics ${METRICS} \
        ${LIMIT:+--limit ${LIMIT}} \
        ${PER_CATEGORY:+--per-category} \
        ${EXTRA_ARGS}
}

# Prints tag, manifest summary and the bank tree.
inspect_tag() {
    local tag="$1"
    if [ ! -d "${BANK_ROOT}/${tag}" ]; then
        echo "No bank tree for tag '${tag}' at ${BANK_ROOT}/${tag}"
        echo "Existing tags:"
        ls -1 "${BANK_ROOT}" 2>/dev/null || echo "  (none)"
        return 1
    fi
    python - "${BANK_ROOT}" "${tag}" "${RESULTS_DIR}" <<'PY'
import json, os, sys

bank_root, tag, results_dir = sys.argv[1], sys.argv[2], sys.argv[3]
root = os.path.join(bank_root, tag)
manifest = os.path.join(root, "manifest.json")

print(f"tag            : {tag}")
print(f"bank tree      : {root}")

if os.path.exists(manifest):
    with open(manifest, encoding="utf-8") as fh:
        m = json.load(fh)
    print(f"phase          : {m.get('phase')}")
    print(f"ingest_config  : {m.get('ingest_config')}")
    print(f"created_at     : {m.get('created_at')}")
    print(f"elapsed_sec    : {m.get('elapsed_sec')}")
    print(f"conversations  : {m.get('n_conversations')}")
    for name, info in (m.get("systems") or {}).items():
        print(f"  {name:<20} notes={info.get('notes', 0):<7} banks={len(info.get('banks') or {})}")
else:
    print("manifest.json  : MISSING (did the ingest phase finish?)")

print("\nbank files (SQLite):")
count = 0
for dirpath, _dirnames, filenames in os.walk(root):
    for name in sorted(filenames):
        if name.endswith(".sqlite"):
            count += 1
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            print(f"  {rel}")
print(f"  total: {count}")

preds_dir = os.path.join(results_dir, "preds")
if os.path.isdir(preds_dir):
    saved = sorted(f for f in os.listdir(preds_dir) if f.startswith(tag + "__"))
    print(f"\nsaved predictions (Phase B, resumed on re-run): {len(saved)}")
    for name in saved:
        print(f"  {name}")

results = []
if os.path.isdir(results_dir):
    results = sorted(
        f for f in os.listdir(results_dir)
        if f.startswith(tag + "__") and not f.endswith(".jsonl")
    )
print(f"\nresults in {results_dir}:")
for name in results:
    print(f"  {name}")
if not results:
    print("  (none yet)")
PY
}

# Deletes saved predictions so Phase B re-answers instead of resuming.
rerun_tag() {
    local tag="$1"
    python - "${RESULTS_DIR}" "${tag}" <<'PY'
import glob, os, sys

results_dir, tag = sys.argv[1], sys.argv[2]
files = glob.glob(os.path.join(results_dir, "preds", f"{tag}__*.jsonl"))
if not files:
    print(f"No saved predictions for tag '{tag}' — nothing to clear.")
for path in files:
    os.remove(path)
    print(f"removed {path}")
print(f"{len(files)} prediction file(s) removed; the next retrieve run starts fresh.")
PY
}

clean_tag() {
    local tag="$1"
    python - "${BANK_ROOT}" "${WORK_ROOT}" "${RESULTS_DIR}" "${tag}" <<'PY'
import glob, os, shutil, sys

bank_root, work_root, results_dir, tag = sys.argv[1:5]
dirs = [os.path.join(bank_root, tag), os.path.join(work_root, tag)]
for path in dirs:
    if os.path.isdir(path):
        shutil.rmtree(path)
        print(f"removed tree  {path}")

files = []
for pattern in (
    os.path.join(results_dir, f"{tag}__*"),
    os.path.join(results_dir, "preds", f"{tag}__*"),
):
    files.extend(glob.glob(pattern))
for path in files:
    if os.path.isfile(path):
        os.remove(path)
        print(f"removed file  {path}")
print(f"cleaned tag '{tag}' ({len(files)} file(s))")
PY
}

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
MODE="${1:-help}"

case "${MODE}" in

env)
    ensure_env
    verify_env
    ;;

smoke)
    log "Smoke test — local HF, tag '${SMOKE_TAG}'"
    ensure_env
    verify_env
    warn "--limit only shrinks Phase B; Phase A still ingests all conversations."
    # Narrow the scope for a fast end-to-end check (this script runs one mode).
    SYSTEMS="FastASEM"
    LIMIT="10"
    phase_ingest "${SMOKE_TAG}" "${HF_INGEST_CONFIG}"
    phase_retrieve "${SMOKE_TAG}" "${HF_INGEST_CONFIG}" ""
    log "Smoke results"
    echo "  banks  : ${BANK_ROOT}/${SMOKE_TAG}"
    echo "  results: ${RESULTS_DIR}/${SMOKE_TAG}__*.json"
    ;;

hf-ingest)
    log "Phase A (local HF) — tag '${HF_TAG}' via ${HF_INGEST_CONFIG}"
    ensure_env
    verify_env
    phase_ingest "${HF_TAG}" "${HF_INGEST_CONFIG}"
    inspect_tag "${HF_TAG}"
    ;;

hf-retrieve)
    log "Phase B (local HF) — tag '${HF_TAG}' group '${HF_GROUP}'"
    ensure_env
    verify_env
    phase_retrieve "${HF_TAG}" "" "${HF_GROUP}"
    ;;

hf-all)
    log "Phase A + B (local HF) — tag '${HF_TAG}'"
    ensure_env
    verify_env
    phase_ingest "${HF_TAG}" "${HF_INGEST_CONFIG}"
    phase_retrieve "${HF_TAG}" "" "${HF_GROUP}"
    ;;

api-ingest)
    log "Phase A (API) — tag '${API_TAG}' via ${API_INGEST_CONFIG}"
    ensure_env
    verify_env
    if [ -z "${OPENAI_BASE_URL:-}" ]; then
        warn "OPENAI_BASE_URL is unset — set it in .env or the API config will fail."
    fi
    phase_ingest "${API_TAG}" "${API_INGEST_CONFIG}"
    inspect_tag "${API_TAG}"
    ;;

api-retrieve)
    log "Phase B (API) — tag '${API_TAG}'"
    ensure_env
    verify_env
    if [ -n "${API_RETRIEVE_CONFIG}" ]; then
        echo "[api] single-model run via --config ${API_RETRIEVE_CONFIG}"
        phase_retrieve "${API_TAG}" "${API_RETRIEVE_CONFIG}" ""
    else
        echo "[api] group '${API_GROUP}' — the endpoint must serve every model name"
        echo "      in that group; otherwise set API_RETRIEVE_CONFIG to a single config."
        phase_retrieve "${API_TAG}" "" "${API_GROUP}"
    fi
    ;;

api-all)
    log "Phase A + B (API) — tag '${API_TAG}'"
    ensure_env
    verify_env
    phase_ingest "${API_TAG}" "${API_INGEST_CONFIG}"
    if [ -n "${API_RETRIEVE_CONFIG}" ]; then
        phase_retrieve "${API_TAG}" "${API_RETRIEVE_CONFIG}" ""
    else
        phase_retrieve "${API_TAG}" "" "${API_GROUP}"
    fi
    ;;

sweep)
    log "Ingest once (local HF) then sweep retrieval over '${ALL_GROUP}'"
    ensure_env
    verify_env
    warn "Group '${ALL_GROUP}' mixes *_hf and *_api configs: the HF runs are local"
    warn "and the API runs need a live endpoint serving those exact model names."
    phase_ingest "${HF_TAG}" "${HF_INGEST_CONFIG}"
    phase_retrieve "${HF_TAG}" "" "${ALL_GROUP}"
    ;;

inspect)
    ensure_env
    inspect_tag "${2:-${HF_TAG}}"
    ;;

rerun)
    ensure_env
    rerun_tag "${2:-${HF_TAG}}"
    ;;

clean)
    ensure_env
    warn "This deletes ingested banks for the tag — Phase A must run again afterwards."
    clean_tag "${2:-${HF_TAG}}"
    ;;

help | *)
    sed -n '2,68p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    ;;

esac
