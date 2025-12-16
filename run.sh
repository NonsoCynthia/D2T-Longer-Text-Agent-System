#!/usr/bin/env bash

# source /home/chinonso/anaconda3/etc/profile.d/conda.sh
# conda activate agent

# Wrapper around run.py
# Edit arguments as needed.

python run.py \
  --languages ga \
  --systems unified_worker \
  --dataset data/D2T-1-FA_same3to6_min8_max199.xml \
  --provider openai \
  --results-dir results \
  --max-reruns 3


  # --systems default unified_worker e2e \

# # Make the shell script executable:
# chmod +x run_d2t_experiments.sh

# # Run all samples, both end to end and pipeline, English
# ./run_d2t_experiments.sh

# # Run only pipeline, Irish, on default dataset
# ./run_d2t_experiments.sh openai default ga pipeline

# # Run samples 1 to 10 with e2e only
# ./run_d2t_experiments.sh openai default en e2e data/D2T-1-FA_same3to6_min50_max500_sample.xml results_cli 1 10

