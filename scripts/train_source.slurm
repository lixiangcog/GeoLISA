#!/bin/bash
#SBATCH --job-name=geolisa-source
#SBATCH --partition=4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-source-%j.out

set -euo pipefail
: "${GEOLISA_ROOT:?Set GEOLISA_ROOT to the repository path}"
: "${GEOLISA_ENV:?Set GEOLISA_ENV to the conda environment path}"
: "${FUNDUS_ROOT:?Set FUNDUS_ROOT to the extracted Fundus directory}"

mkdir -p "${GEOLISA_ROOT}/logs"
cd "${GEOLISA_ROOT}/OPTIC"
"${GEOLISA_ENV}/bin/python" train_source.py \
  --dataset-root "${FUNDUS_ROOT}" \
  --source-dataset RIM_ONE_r3 \
  --path-save-model "${GEOLISA_ROOT}/models" \
  --path-save-log "${GEOLISA_ROOT}/logs" \
  --num-epochs "${SOURCE_EPOCHS:-200}" \
  --batch-size 8 --num-workers 8 --device cuda:0
