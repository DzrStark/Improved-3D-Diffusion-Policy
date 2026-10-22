#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TRAIN_PYTHON="${PYTHON_BIN:-python}"

ZARR_PATH="../tsingtec_training_data/right_pick"
GPU=0
SEED=42
OUTPUT_DIR=""

usage() {
    cat <<'EOF'
Usage: tools/train_g1_dex3_rgb.sh [options]

Options:
  --zarr PATH        Training Zarr (default: ../tsingtec_training_data/right_pick)
  --gpu INDEX        CUDA device index (default: 0)
  --seed INTEGER     Random seed (default: 42)
  --output-dir PATH  Run directory (default: unique path under data/outputs/g1_dex3_rgb)
  --python PATH      Python interpreter (default: PYTHON_BIN or python on PATH)
  -h, --help         Show this help

Run from any directory; relative paths are resolved from the project root.
WandB logging runs online. All 60 episodes are used, with checkpoint saving enabled.
Arguments after -- are passed directly to Hydra (e.g. -- --cfg job --resolve).
EOF
}

while (($#)); do
    case "$1" in
        --zarr) ZARR_PATH="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --python) TRAIN_PYTHON="$2"; shift 2 ;;
        --) shift; break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$ZARR_PATH" != /* ]]; then
    ZARR_PATH="${PROJECT_ROOT}/${ZARR_PATH}"
fi
if [[ -z "$OUTPUT_DIR" ]]; then
    RUN_DATE="$(date +%Y.%m.%d)"
    RUN_TIME="$(date +%H.%M.%S.%N)"
    OUTPUT_DIR="${PROJECT_ROOT}/data/outputs/g1_dex3_rgb/${RUN_DATE}/${RUN_TIME}_seed${SEED}"
elif [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_DIR}"
fi

if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Output already exists; choose a new --output-dir: $OUTPUT_DIR" >&2
    exit 2
fi

cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
# Previous logging override: "logging.mode=offline"
# Use online logging as requested; the YAML also defaults to online.
exec "$TRAIN_PYTHON" train.py --config-name=dp_g1_dex3_rgb \
    "task.dataset.zarr_path='${ZARR_PATH}'" \
    "training.device=cuda:0" \
    "training.seed=${SEED}" \
    "logging.mode=online" \
    "hydra.run.dir='${OUTPUT_DIR}'" \
    "multi_run.run_dir='${OUTPUT_DIR}'" \
    "$@"
