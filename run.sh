#!/usr/bin/env bash

source /home/chinonso/anaconda3/etc/profile.d/conda.sh
conda activate lang2

set -e

PROVIDER="${1:-openai}"
WORKFLOW="${2:-default}"
LANGUAGE="${3:-en}"
RUN_MODE="${4:-both}"
DATASET_PATH="${5:-data/D2T-1-FA_same3to6_min50_max500_sample.xml}"
OUTPUT_DIR="${6:-results_cli}"
START_ID="${7:-1}"
END_ID="${8:-}"

CMD=(python run_d2t_experiments.py
  --provider "$PROVIDER"
  --workflow "$WORKFLOW"
  --language "$LANGUAGE"
  --run-mode "$RUN_MODE"
  --dataset-path "$DATASET_PATH"
  --output-dir "$OUTPUT_DIR"
  --start-id "$START_ID"
)

if [ -n "$END_ID" ]; then
  CMD+=(--end-id "$END_ID")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"




# # Make the shell script executable:
# chmod +x run_d2t_experiments.sh

# # Run all samples, both end to end and pipeline, English
# ./run_d2t_experiments.sh

# # Run only pipeline, Irish, on default dataset
# ./run_d2t_experiments.sh openai default ga pipeline

# # Run samples 1 to 10 with e2e only
# ./run_d2t_experiments.sh openai default en e2e data/D2T-1-FA_same3to6_min50_max500_sample.xml results_cli 1 10

