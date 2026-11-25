#!/usr/bin/env bash
# Steps 8-9 driver: within-slide-negative heads (B/C linear, D/E MLP), 5-fold CV
# + Step-7 report. Also the entry point of the Kubernetes job (submit_mlp_job.py).
#
#   bash scripts/huncrc/run_cv.sh train    # all (variant, fold) runs, resumable
#   bash scripts/huncrc/run_cv.sh report   # report/table from whatever is finished
#   VARIANTS="D E" bash scripts/huncrc/run_cv.sh train   # Step 9 MLP
#
# Every path can be overridden through the environment (defaults below). The
# feature caches are (re)built only if missing, so the script also works after a
# restart that cleared them. test_final is never read; nothing goes to MLflow.
set -euo pipefail

cmd="${1:?usage: run_cv.sh train|report}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PRE="${PRE:-/mnt/projects/huncrc_dysplasia/preprocessing}"
SLIDE_CSV="${SLIDE_CSV:-/mnt/data/Public/Public/colorectum/TCIA_HunCRC/slide_level_annotations.csv}"
# Results and caches live on the project share, never in the home directory.
SHARE_TC="/mnt/projects/huncrc_dysplasia/tile_classifier"
CACHE_ROOT="${CACHE_ROOT:-$SHARE_TC/cache}"
OUT="${OUT:?set OUT to a new results dir, e.g. OUT=$SHARE_TC/<step>; train resumes in an existing one}"
A_RUNS="${A_RUNS-}"                       # report: Step-4a runs dir; "" or missing dir = skip A
OTHER_RUNS="${OTHER_RUNS:-}"              # report: more runs dirs, e.g. the Step-8 B/C runs
EPOCHS="${EPOCHS:-10}"
NEG_CAP="${NEG_CAP:-5000}"
VARIANTS="${VARIANTS:-B C}"
FOLDS="${FOLDS:-0 1 2 3 4}"
N_BOOT="${N_BOOT:-1000}"
SMOKE_TILES="${SMOKE_TILES:-}"            # e.g. 300 for a quick end-to-end check
cgroup_cpus() {  # CPU quota of this container; nproc shows the node's cores in a pod
  local q p
  if read -r q p < /sys/fs/cgroup/cpu.max 2>/dev/null && [[ "$q" != max ]]; then
    echo $(( (q + p - 1) / p ))                                   # cgroup v2
  elif q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null) && (( q > 0 )); then
    p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)                 # cgroup v1
    echo $(( (q + p - 1) / p ))
  else
    nproc
  fi
}
THREADS="${THREADS:-$(cgroup_cpus)}"

export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PY="$REPO/.venv/bin/python"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

mkdir -p "$OUT" "$CACHE_ROOT"
cd "$REPO"
echo "[$(ts)] run_cv.sh $cmd: threads=$THREADS out=$OUT cache_root=$CACHE_ROOT git=$(git rev-parse --short HEAD)$(git diff --quiet -- ml scripts || echo '+dirty')"
echo "[$(ts)] host=$(hostname) nproc=$(nproc) cgroup cpu.max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a) memory.max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo n/a) variants=$VARIANTS folds=$FOLDS epochs=$EPOCHS"

common=(--tiling "$PRE/tiling_200" --slide-csv "$SLIDE_CSV")
build() {  # prints the cache dir; reuses an existing cache with the same key
  "$PY" scripts/huncrc/build_feature_cache.py "${common[@]}" \
    --embeddings "$PRE/embeddings/prov-gigapath" --cache-root "$CACHE_ROOT" "$@" | head -1
}
TRAIN_CACHE="$(build --splits train --within-slide-negatives --exclude-carcinoma --annotations "$PRE/annotations")"
PRELIM_ALL="$(build --splits test_preliminary --mode all)"
echo "[$(ts)] train cache:  $TRAIN_CACHE"
echo "[$(ts)] prelim cache: $PRELIM_ALL"

args=(--train-cache "$TRAIN_CACHE" --prelim-all-cache "$PRELIM_ALL" --runs "$OUT/runs" "${common[@]}" --threads "$THREADS")
case "$cmd" in
  train)
    smoke=()
    [[ -n "$SMOKE_TILES" ]] && smoke=(--smoke-tiles-per-slide "$SMOKE_TILES")
    # shellcheck disable=SC2086  # VARIANTS / FOLDS are space-separated lists
    "$PY" scripts/huncrc/within_slide_cv.py train "${args[@]}" --variants $VARIANTS --folds $FOLDS \
      --epochs "$EPOCHS" --neg-cap "$NEG_CAP" "${smoke[@]}"
    ;;
  report)
    extra=()
    if [[ -n "$A_RUNS" && -d "$A_RUNS" ]]; then extra+=(--a-runs "$A_RUNS"); else echo "[$(ts)] no Step-4a runs at '$A_RUNS', A skipped"; fi
    # shellcheck disable=SC2086  # OTHER_RUNS is a space-separated list
    [[ -n "$OTHER_RUNS" ]] && extra+=(--other-runs $OTHER_RUNS)
    "$PY" scripts/huncrc/within_slide_cv.py report "${args[@]}" "${extra[@]}" \
      --annotations "$PRE/annotations" --n-boot "$N_BOOT" --out "$OUT"
    ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
echo "[$(ts)] run_cv.sh $cmd: finished"
