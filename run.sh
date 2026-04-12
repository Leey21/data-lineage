#!/bin/bash
#
# Data lineage analysis run script
#
# Configure environment variables before use (replace with your own values):
#
#   export OPENAI_BASE_URL="https://your-api-proxy.com/v1"
#   export OPENAI_API_KEY="sk-your-api-key"
#   export HUGGINGFACE_API_TOKEN="hf_your-token"   # optional
#
# Example invocations:
#
#   bash run.sh                              # default: use datasets.txt in text-only mode
#   bash run.sh datasets.txt
#   bash run.sh datasets.txt --output-dir output
#   bash run.sh datasets.txt --max-depth 3
#   bash run.sh datasets.txt --multimodal true
#   bash run.sh datasets.txt --multimodal true --log-level DEBUG
#

set -e
ORIGINAL_CWD="$(pwd)"

print_help() {
    echo "Usage: $0 <dataset_file> [options]"
    echo ""
    echo "Positional:"
    echo "  dataset_file          File with dataset names (one per line)"
    echo ""
    echo "Model (each agent can use a different model):"
    echo "  --model NAME          Default model for all agents (default: gpt-5.4)"
    echo "  --model-sourcing      SourcingAgent: extract links from README"
    echo "  --model-tracing       TracingAgent: trace datasets from README/blog/GitHub"
    echo "  --model-paper         PaperAgent: analyze paper content"
    echo "  --model-classification ClassificationAgent: classify dataset type"
    echo "  --model-aggregation   AggregationAgent: infer missing dataset names"
    echo "  --model-dataset-builder DatasetBuilder: README summary, data type"
    echo ""
    echo "Other:"
    echo "  --output-dir DIR      Output directory (default: output, resolved under data_lineage/)"
    echo "  --max-depth N         Recursion depth, -1=unlimited (default: -1, trace to bottom)"
    echo "  --log-level LEVEL     DEBUG|INFO|WARNING|ERROR (default: INFO)"
    echo "  --multimodal BOOL     Use multimodal classification prompt (true/false, default: false)"
    echo "                         Set --multimodal true for multimodal dataset analysis"
    echo "  --no-load-existing    Start fresh, do not load existing results"
    echo "  --blog-analysis       Enable blog analysis (default: on)"
    echo "  --no-blog-analysis    Disable blog analysis"
    echo "  --paper-analysis      Enable paper analysis (default: on)"
    echo "  --no-paper-analysis   Disable paper analysis"
    echo "  --pdf-crop            Crop PDF by TOC (intro~conclusion); default: off, use full PDF"
    echo ""
    echo "Environment (replace with your own values):"
    echo "  OPENAI_BASE_URL         API relay base URL (required)"
    echo "  OPENAI_API_KEY          API key (required)"
    echo "  HUGGINGFACE_API_TOKEN   HF token (optional)"
}

for arg in "$@"; do
    if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
        print_help
        exit 0
    fi
done

if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
    echo "Error: OPENAI_BASE_URL is not set"
    exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Error: OPENAI_API_KEY is not set"
    exit 1
fi

# =============================================================================
# Default configuration
# =============================================================================
DEFAULT_MODEL="gpt-5.4"

# Per-agent models (override with env or args)
MODEL_SOURCING="${MODEL_SOURCING:-$DEFAULT_MODEL}"      # SourcingAgent: link extraction from README
MODEL_TRACING="${MODEL_TRACING:-$DEFAULT_MODEL}"        # TracingAgent: trace from README/blog/GitHub
MODEL_PAPER="${MODEL_PAPER:-$DEFAULT_MODEL}"            # PaperAgent: paper content analysis
MODEL_CLASSIFICATION="${MODEL_CLASSIFICATION:-$DEFAULT_MODEL}"  # ClassificationAgent: dataset classification
MODEL_AGGREGATION="${MODEL_AGGREGATION:-$DEFAULT_MODEL}"        # AggregationAgent: infer missing dataset names
MODEL_DATASET_BUILDER="${MODEL_DATASET_BUILDER:-$DEFAULT_MODEL}"  # DatasetBuilder: README summary, data type

# OUTPUT_DIR default set after SCRIPT_DIR (so output goes to this package's output directory by default)
MAX_DEPTH="${MAX_DEPTH:--1}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
MULTIMODAL="${MULTIMODAL:-false}"

# ============================================================================= 
# Parse Arguments
# =============================================================================
DATASET_FILE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            DEFAULT_MODEL="$2"
            MODEL_SOURCING="$2"
            MODEL_TRACING="$2"
            MODEL_PAPER="$2"
            MODEL_CLASSIFICATION="$2"
            MODEL_AGGREGATION="$2"
            MODEL_DATASET_BUILDER="$2"
            shift 2
            ;;
        --model-sourcing)
            MODEL_SOURCING="$2"
            shift 2
            ;;
        --model-tracing)
            MODEL_TRACING="$2"
            shift 2
            ;;
        --model-paper)
            MODEL_PAPER="$2"
            shift 2
            ;;
        --model-classification)
            MODEL_CLASSIFICATION="$2"
            shift 2
            ;;
        --model-aggregation)
            MODEL_AGGREGATION="$2"
            shift 2
            ;;
        --model-dataset-builder)
            MODEL_DATASET_BUILDER="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-depth)
            MAX_DEPTH="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --multimodal)
            MULTIMODAL="$2"
            shift 2
            ;;
        --no-load-existing)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        --blog-analysis)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        --no-blog-analysis)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        --paper-analysis)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        --no-paper-analysis)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        --pdf-crop)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        -*)
            EXTRA_ARGS+=("$1")
            if [[ "$1" == *=* ]] || [[ $# -eq 1 ]]; then
                shift
            else
                EXTRA_ARGS+=("$2")
                shift 2
            fi
            ;;
        *)
            if [[ -z "$DATASET_FILE" ]]; then
                DATASET_FILE="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

if [[ -z "$DATASET_FILE" ]]; then
    DATASET_FILE="datasets.txt"
    DATASET_FILE_IS_DEFAULT=true
else
    DATASET_FILE_IS_DEFAULT=false
fi

# =============================================================================
# Run
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RAW_DATASET_FILE="$DATASET_FILE"
if ! DATASET_FILE="$(python - "$RAW_DATASET_FILE" "$DATASET_FILE_IS_DEFAULT" "$ORIGINAL_CWD" "$SCRIPT_DIR" "$PROJECT_ROOT" <<'PY'
import os
import sys

raw_path, is_default, original_cwd, script_dir, project_root = sys.argv[1:]

if is_default == "true":
    candidate = os.path.join(script_dir, raw_path)
elif os.path.isabs(raw_path):
    candidate = raw_path
else:
    candidate = os.path.join(original_cwd, raw_path)

candidate = os.path.realpath(candidate)
if not os.path.isfile(candidate):
    sys.exit(1)

print(os.path.relpath(candidate, project_root))
PY
)"; then
    echo "Error: dataset file not found: $RAW_DATASET_FILE"
    exit 1
fi
OUTPUT_DIR="${OUTPUT_DIR:-output}"
# Run from parent dir so "python -m data_lineage" works
cd "$PROJECT_ROOT"

echo "=============================================="
echo "Data Lineage Analysis"
echo "=============================================="
echo "Dataset file: $DATASET_FILE"
echo "Output dir:   $OUTPUT_DIR"
echo "Max depth:    $MAX_DEPTH"
echo "Multimodal:   $MULTIMODAL"
echo ""
echo "Model configuration:"
echo "  sourcing:        $MODEL_SOURCING"
echo "  tracing:         $MODEL_TRACING"
echo "  paper:           $MODEL_PAPER"
echo "  classification:  $MODEL_CLASSIFICATION"
echo "  aggregation:     $MODEL_AGGREGATION"
echo "  dataset_builder: $MODEL_DATASET_BUILDER"
echo "=============================================="

python -m data_lineage "$DATASET_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --max-depth "$MAX_DEPTH" \
    --log-level "$LOG_LEVEL" \
    --multimodal "$MULTIMODAL" \
    --model "$DEFAULT_MODEL" \
    --model-sourcing "$MODEL_SOURCING" \
    --model-tracing "$MODEL_TRACING" \
    --model-paper "$MODEL_PAPER" \
    --model-classification "$MODEL_CLASSIFICATION" \
    --model-aggregation "$MODEL_AGGREGATION" \
    --model-dataset-builder "$MODEL_DATASET_BUILDER" \
    "${EXTRA_ARGS[@]}"
