#!/bin/bash
#SBATCH --job-name=geolisa-ctta
#SBATCH --partition=4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-geolisa-%j.out

set -euo pipefail
: "${GEOLISA_ROOT:?Set GEOLISA_ROOT to the repository path}"
: "${GEOLISA_ENV:?Set GEOLISA_ENV to the conda environment path}"
: "${FUNDUS_ROOT:?Set FUNDUS_ROOT to the extracted Fundus directory}"

mkdir -p "${GEOLISA_ROOT}/logs" "${GEOLISA_ROOT}/results"
cd "${GEOLISA_ROOT}/OPTIC"
"${GEOLISA_ENV}/bin/python" lisa.py \
  --dataset-root "${FUNDUS_ROOT}" \
  --model-root "${GEOLISA_ROOT}/models" \
  --source-dataset RIM_ONE_r3 \
  --target-datasets Drishti_GS \
  --text-embeddings "${GEOLISA_ROOT}/models/biomedclip_optic_text.pt" \
  --semantic-backend sam \
  --sam-checkpoint "${GEOLISA_ROOT}/models/sam_vit_b_01ec64.pth" \
  --prior-path "${GEOLISA_ROOT}/models/RIM_ONE_r3/dgfa_prior.pt" \
  --output "${GEOLISA_ROOT}/results/rim_to_drishti.json" \
  --device cuda:0
