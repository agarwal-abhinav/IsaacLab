#!/bin/bash
# Collect demonstrations with the NON-MARKOVIAN scripted gear-assembly expert.
# Modelled on FurnitureSim's src/data_collection/collect_scripted.sh.
#
# Usage (from the IsaacLab repo root, with the Isaac Lab env active):
#   ./scripts/tools/collect_scripted_nm.sh [n_demos] [num_envs] [suffix]
#
#   n_demos  (default 200): target total successful demos in the output file
#   num_envs (default 16) : parallel envs (headless)
#   suffix   (default "") : appended to the dataset name, e.g. "v2" ->
#                           gear_assembly_scripted_non_markovian_v2.hdf5
#
# Extra env vars:
#   DATASET_DIR (default ./datasets), N_VIDEO (default 20), SEED (optional),
#   MAX_STEPS (default: the script's NM default, 450).
#
# Hidden variables (latent plan, alignment cycles, sticky delays) of every
# episode are written to <dataset>_nm_latents.jsonl and, for successful demos,
# attached as the JSON attr `nm_latents` on each /data/demo_N group.

N_DEMOS=${1:-200}
N_ENVS=${2:-16}
SUFFIX=${3:-""}
DATASET_DIR=${DATASET_DIR:-./datasets}
N_VIDEO=${N_VIDEO:-20}

NAME="gear_assembly_scripted_non_markovian$( [ -n "${SUFFIX}" ] && echo "_${SUFFIX}" )"
DATASET_FILE="${DATASET_DIR}/${NAME}.hdf5"

echo "N_DEMOS: ${N_DEMOS}"
echo "N_ENVS: ${N_ENVS}"
echo "DATASET_FILE: ${DATASET_FILE}"

EXTRA=""
[ -n "${SEED}" ] && EXTRA="${EXTRA} --seed ${SEED}"
[ -n "${MAX_STEPS}" ] && EXTRA="${EXTRA} --max_steps_per_demo ${MAX_STEPS}"

# Line-buffered output so log lines survive an abrupt exit.
stdbuf -oL -eL ./isaaclab.sh -p scripts/tools/collect_demos_scripted.py \
    --task Isaac-GearAssembly-Franka-IK-Rel-Mimic-v0 \
    --non_markovian \
    --dataset_file "${DATASET_FILE}" \
    --num_envs "${N_ENVS}" \
    --num_demos "${N_DEMOS}" \
    --n-video-trials "${N_VIDEO}" \
    --enable_cameras --headless ${EXTRA}
