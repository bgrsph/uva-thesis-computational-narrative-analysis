#!/usr/bin/env bash
# train_bert_crf_on_maven.sh
#
# Reproducible end-to-end setup + training for the BERT+CRF event detector
# on MAVEN. Mirrors section 1.2 of notebooks/pipeline.ipynb.
#
# Pinning rationale (do not change without reading the notebook):
#   - venv lives in models/bert_crf/.venv-maven-train (colocated with code)
#   - Python must be 3.9.x; transformers==4.18.0 has no wheels for >=3.11
#   - torch / transformers / tokenizers pinned for arm64 wheel availability
#
# Usage (from anywhere):
#   bash models/bert_crf/train_bert_crf_on_maven.sh            # setup + train + eval
#   bash models/bert_crf/train_bert_crf_on_maven.sh --eval-only # reuse existing checkpoint
#   bash models/bert_crf/train_bert_crf_on_maven.sh --setup-only
#
# All stdout/stderr is tee'd to data/intermediate/models/bert_crf/logs/<timestamp>.log
#
# Runs under: macOS (bash/zsh), Linux, Windows via Git Bash / MSYS2 / WSL.
# Native Windows cmd/PowerShell is NOT supported; use Git Bash or WSL.

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
MODE="all"  # all | setup-only | eval-only
for arg in "$@"; do
    case "$arg" in
        --setup-only) MODE="setup-only" ;;
        --eval-only)  MODE="eval-only" ;;
        -h|--help)
            sed -n '2,22p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve paths (portable: no `readlink -f`, no GNU-only flags)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
VENV_DIR="$SCRIPT_DIR/.venv-maven-train"
DATA_DIR="$REPO_ROOT/data/raw/MAVEN"
OUTPUT_DIR="$REPO_ROOT/data/intermediate/models/bert_crf"
LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

# Tee every subsequent line (stdout + stderr) to the log
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===================================================================="
echo " BERT+CRF on MAVEN — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " mode:       $MODE"
echo " script dir: $SCRIPT_DIR"
echo " repo root:  $REPO_ROOT"
echo " data dir:   $DATA_DIR"
echo " output dir: $OUTPUT_DIR"
echo " log file:   $LOG_FILE"
echo "===================================================================="

# ---------------------------------------------------------------------------
# OS detection
# ---------------------------------------------------------------------------
UNAME_S="$(uname -s 2>/dev/null || echo Unknown)"
case "$UNAME_S" in
    Darwin*)                    OS="macos" ;;
    Linux*)                     OS="linux" ;;
    MINGW*|MSYS*|CYGWIN*)       OS="windows" ;;
    *)                          OS="unknown" ;;
esac
echo "[os] detected: $OS ($UNAME_S)"

if [ "$OS" = "unknown" ]; then
    echo "[os] unrecognized platform; proceeding with POSIX defaults" >&2
fi

# Activation script path differs on Windows (Scripts/) vs POSIX (bin/)
if [ "$OS" = "windows" ]; then
    VENV_ACTIVATE="$VENV_DIR/Scripts/activate"
    VENV_PY="$VENV_DIR/Scripts/python.exe"
else
    VENV_ACTIVATE="$VENV_DIR/bin/activate"
    VENV_PY="$VENV_DIR/bin/python"
fi

# ---------------------------------------------------------------------------
# Locate a Python 3.9 interpreter
# ---------------------------------------------------------------------------
find_python39() {
    # Preferred order:
    #   1. /usr/bin/python3 (Apple CommandLineTools on macOS ships 3.9.6)
    #   2. python3.9 on PATH
    #   3. python3 on PATH (only accepted if it reports 3.9.x)
    #   4. py -3.9 (Windows launcher, if available)
    local candidates=()
    if [ "$OS" = "macos" ] && [ -x "/usr/bin/python3" ]; then
        candidates+=("/usr/bin/python3")
    fi
    candidates+=("python3.9" "python3")
    if [ "$OS" = "windows" ] && command -v py >/dev/null 2>&1; then
        # `py -3.9` is a launcher invocation, not a single binary — handle specially
        if py -3.9 -c "import sys; assert sys.version_info[:2]==(3,9)" >/dev/null 2>&1; then
            echo "py -3.9"
            return 0
        fi
    fi

    local c v
    for c in "${candidates[@]}"; do
        if command -v "$c" >/dev/null 2>&1; then
            v="$("$c" -c 'import sys; print("%d.%d"%sys.version_info[:2])' 2>/dev/null || true)"
            if [ "$v" = "3.9" ]; then
                echo "$c"
                return 0
            fi
        fi
    done
    return 1
}

setup_venv() {
    if [ -x "$VENV_PY" ]; then
        local existing_v
        existing_v="$("$VENV_PY" -c 'import sys; print("%d.%d"%sys.version_info[:2])' 2>/dev/null || true)"
        if [ "$existing_v" = "3.9" ]; then
            echo "[venv] reusing existing venv at $VENV_DIR (python $existing_v)"
            return 0
        else
            echo "[venv] existing venv reports python '$existing_v' (want 3.9); removing and recreating"
            rm -rf "$VENV_DIR"
        fi
    fi

    echo "[venv] locating a Python 3.9 interpreter..."
    local py39
    if ! py39="$(find_python39)"; then
        cat >&2 <<EOF
[venv] ERROR: no Python 3.9 interpreter found.

  transformers==4.18.0 has no wheels for Python >= 3.11. You must install
  Python 3.9 before running this script.

  macOS:    xcode-select --install     # ships /usr/bin/python3 == 3.9.x
            or: brew install python@3.9
  Linux:    sudo apt-get install python3.9 python3.9-venv
            or use pyenv: pyenv install 3.9.19
  Windows:  https://www.python.org/downloads/release/python-3919/
            (install with "py launcher" checked, then re-run under Git Bash)
EOF
        exit 1
    fi
    echo "[venv] using: $py39"

    # `py -3.9` needs to stay unquoted-as-a-whole; eval handles both cases
    eval "$py39 -m venv \"$VENV_DIR\""

    # Verify
    local v
    v="$("$VENV_PY" -c 'import sys; print("%d.%d.%d"%sys.version_info[:3])')"
    case "$v" in
        3.9.*) echo "[venv] created with python $v" ;;
        *)
            echo "[venv] ERROR: venv has python $v, expected 3.9.x" >&2
            exit 1
            ;;
    esac
}

install_deps() {
    # shellcheck disable=SC1090
    . "$VENV_ACTIVATE"
    python -V
    python -m pip install --upgrade pip
    # Pin only what the notebook pins; let pip resolve a torch wheel that matches
    # the platform + python version (macOS arm64, linux x86_64, etc.).
    python -m pip install \
        torch \
        transformers==4.18.0 \
        tokenizers==0.11.6 \
        seqeval \
        scikit-learn \
        numpy
    echo "[deps] installed"
}

# ---------------------------------------------------------------------------
# Dataset sanity check
# ---------------------------------------------------------------------------
check_data() {
    local f
    for f in train.jsonl valid.jsonl test.jsonl; do
        if [ ! -f "$DATA_DIR/$f" ]; then
            echo "[data] ERROR: missing $DATA_DIR/$f" >&2
            echo "[data] place the MAVEN jsonl files under $DATA_DIR and re-run." >&2
            exit 1
        fi
    done
    echo "[data] MAVEN files present"
}

# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
run_train() {
    # shellcheck disable=SC1090
    . "$VENV_ACTIVATE"
    cd "$SCRIPT_DIR"
    echo "[train] starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    python run_maven.py \
        --data_dir "$DATA_DIR" \
        --model_type bertcrf \
        --model_name_or_path bert-base-uncased \
        --output_dir "$OUTPUT_DIR" \
        --max_seq_length 128 \
        --do_lower_case \
        --per_gpu_train_batch_size 16 \
        --per_gpu_eval_batch_size 16 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-5 \
        --num_train_epochs 5 \
        --save_steps 100 \
        --logging_steps 100 \
        --seed 0 \
        --do_train \
        --do_eval \
        --evaluate_during_training \
        --overwrite_output_dir
    echo "[train] finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

run_eval_only() {
    # shellcheck disable=SC1090
    . "$VENV_ACTIVATE"
    cd "$SCRIPT_DIR"
    echo "[eval] starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    python run_maven.py \
        --data_dir "$DATA_DIR" \
        --model_type bertcrf \
        --model_name_or_path bert-base-uncased \
        --output_dir "$OUTPUT_DIR" \
        --max_seq_length 128 \
        --do_lower_case \
        --per_gpu_eval_batch_size 16 \
        --seed 0 \
        --do_eval
    echo "[eval] finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$MODE" in
    setup-only)
        setup_venv
        install_deps
        echo "[done] setup complete. activate with: source $VENV_ACTIVATE"
        ;;
    eval-only)
        setup_venv
        install_deps
        check_data
        run_eval_only
        echo "[done] eval complete. results: $OUTPUT_DIR/eval_results.txt"
        ;;
    all)
        setup_venv
        install_deps
        check_data
        run_train
        echo "[done] training + eval complete."
        echo "[done] checkpoint: $OUTPUT_DIR"
        echo "[done] log:        $LOG_FILE"
        ;;
esac
