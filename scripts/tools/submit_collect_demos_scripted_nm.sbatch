#!/bin/bash
#SBATCH --job-name=isaaclab_collect_nm
#SBATCH --account=locomotion
#SBATCH --partition=vision-shared-titanrtx
#SBATCH --exclude=freeman-titanrtx-2,freeman-titanrtx-1,gpu20-2.drl
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH --requeue

# Large-scale data generation with the NON-MARKOVIAN scripted gear expert.
# Same pipeline as submit_collect_demos_scripted.sbatch (Markovian), with
# --non_markovian added, plus optional sharding over a SLURM array.
#
# Single job (one HDF5, then zarr):
#   sbatch scripts/tools/submit_collect_demos_scripted_nm.sbatch
#
# Sharded (e.g. 10 shards x 100 demos, different seeds), then merge + zarr:
#   NUM_DEMOS=100 sbatch --array=0-9 scripts/tools/submit_collect_demos_scripted_nm.sbatch
#   MERGE_SHARDS=1 sbatch scripts/tools/submit_collect_demos_scripted_nm.sbatch
#
# Every setting below can be overridden from the environment, e.g.
#   NUM_DEMOS=1000 NUM_ENVS=32 SEED=7 sbatch scripts/tools/submit_collect_demos_scripted_nm.sbatch
#
# Outputs (NAME defaults to gear_assembly_scripted_nm):
#   datasets/NAME.hdf5 (+ .zarr)        successful demos only; each /data/demo_N
#                                        carries a JSON attr `nm_latents` (hidden plan)
#   datasets/NAME_nm_latents.jsonl       hidden variables of every attempted episode
#   datasets/scripted_videos/            MP4s of the first N_VIDEO attempts
#   datasets/shards/NAME_shardXXX/       per-shard HDF5, latents and videos (array mode)
#   datasets/NAME_videos/                (RENDER_DEMOS=1) MP4 of every saved demo,
#                                        rendered from the recorded camera obs

# ============================================================================
# Parameters (env-overridable)
# ============================================================================
TASK_ID="${TASK_ID:-Isaac-GearAssembly-Franka-IK-Rel-Mimic-v0}"
NAME="${NAME:-gear_assembly_scripted_nm}"
DATASET_DIR="${DATASET_DIR:-./datasets}"
NUM_DEMOS="${NUM_DEMOS:-200}"          # successful demos per job / per shard
NUM_ENVS="${NUM_ENVS:-16}"
MAX_STEPS_PER_DEMO="${MAX_STEPS_PER_DEMO:-450}"   # NM episodes are longer (~300 steps)
SEED="${SEED:-0}"                      # base seed; shard i uses SEED + i
N_VIDEO="${N_VIDEO:-10}"               # MP4s of the first N attempts
RENDER_DEMOS="${RENDER_DEMOS:-0}"      # 1 = also render every saved demo to MP4
NO_NOISE="${NO_NOISE:-0}"              # 1 = zero Markovian-style noise (NM latents remain)
FRESH_COLLECTION="${FRESH_COLLECTION:-1}"
SKIP_COLLECTION="${SKIP_COLLECTION:-0}"
MERGE_SHARDS="${MERGE_SHARDS:-0}"      # 1 = merge NAME_shard*.hdf5 -> NAME.hdf5, then zarr
# ============================================================================

# --- Environment setup (cluster-specific: edit these two lines) ---
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
source /data/locomotion/michzeng/miniconda3/etc/profile.d/conda.sh
conda activate /data/locomotion/michzeng/conda_envs/IsaacLab
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-0}}}"
export PYTHONUNBUFFERED=1
export HOME=/data/locomotion/michzeng
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
mkdir -p logs "$DATASET_DIR"

if [[ -n "${SLURM_ARRAY_TASK_ID}" ]]; then
    SHARD=${SLURM_ARRAY_TASK_ID}
    RUN_SEED=$((SEED + SHARD))
    SHARD_NAME="${NAME}_shard$(printf %03d "$SHARD")"
    # own subdir per shard so scripted_videos/ do not collide
    DATASET_HDF5="${DATASET_DIR}/shards/${SHARD_NAME}/${SHARD_NAME}.hdf5"
    mkdir -p "$(dirname "$DATASET_HDF5")"
    CONVERT=0           # shards are merged and converted by the MERGE_SHARDS job
else
    RUN_SEED=$SEED
    DATASET_HDF5="${DATASET_DIR}/${NAME}.hdf5"
    CONVERT=1
fi
DATASET_ZARR="${DATASET_DIR}/${NAME}.zarr"

echo "DATE: $(date)  NODE: $HOSTNAME  JOB: ${SLURM_JOB_ID} ARRAY_TASK: ${SLURM_ARRAY_TASK_ID:-none}"
echo "TASK_ID=$TASK_ID NUM_DEMOS=$NUM_DEMOS NUM_ENVS=$NUM_ENVS MAX_STEPS=$MAX_STEPS_PER_DEMO SEED=$RUN_SEED"
echo "DATASET_HDF5=$DATASET_HDF5"

# --- Merge-only mode ---
# (merge only shards that finished: a shard is written incrementally, so check
#  each shard log ends with "Done." before merging)
if [[ "$MERGE_SHARDS" == "1" ]]; then
    shopt -s nullglob
    SHARDS=("${DATASET_DIR}/shards/${NAME}"_shard*/"${NAME}"_shard*[0-9].hdf5)
    [[ ${#SHARDS[@]} -eq 0 ]] && { echo "No shards found"; exit 1; }
    echo ">>> Merging ${#SHARDS[@]} shards -> ${DATASET_DIR}/${NAME}.hdf5"
    python scripts/tools/merge_hdf5_datasets.py --input_files "${SHARDS[@]}" \
        --output_file "${DATASET_DIR}/${NAME}.hdf5" || exit 1
    cat "${DATASET_DIR}/shards/${NAME}"_shard*/*_nm_latents.jsonl > "${DATASET_DIR}/${NAME}_nm_latents.jsonl" 2>/dev/null
    DATASET_HDF5="${DATASET_DIR}/${NAME}.hdf5"
    SKIP_COLLECTION=1
    CONVERT=1
fi

# --- Step 1: Collect ---
if [[ "$SKIP_COLLECTION" != "1" ]]; then
    [[ "$FRESH_COLLECTION" == "1" && -f "$DATASET_HDF5" ]] && rm -f "$DATASET_HDF5"
    EXTRA=""
    [[ "$NO_NOISE" == "1" ]] && EXTRA="--no_noise"
    python scripts/tools/collect_demos_scripted.py \
        --task "$TASK_ID" \
        --non_markovian \
        --dataset_file "$DATASET_HDF5" \
        --num_envs "$NUM_ENVS" \
        --num_demos "$NUM_DEMOS" \
        --max_steps_per_demo "$MAX_STEPS_PER_DEMO" \
        --seed "$RUN_SEED" \
        --n-video-trials "$N_VIDEO" \
        --enable_cameras --headless $EXTRA || exit 1
fi

# --- Step 2: Render saved demos (optional) ---
if [[ "$RENDER_DEMOS" == "1" && -f "$DATASET_HDF5" ]]; then
    OUT="${DATASET_HDF5%.hdf5}_videos"
    echo ">>> Rendering demos from camera obs -> $OUT"
    python scripts/tools/hdf5_to_mp4.py --input_file "$DATASET_HDF5" --output_dir "$OUT" \
        --input_keys scene_cam_front scene_cam_rear_left wrist_cam
fi

# --- Step 3: Convert HDF5 -> zarr ---
if [[ "$CONVERT" == "1" ]]; then
    [[ -f "$DATASET_HDF5" ]] || { echo "Dataset HDF5 not found: $DATASET_HDF5"; exit 1; }
    echo ">>> Converting $DATASET_HDF5 -> $DATASET_ZARR"
    python scripts/tools/hdf5_to_zarr.py "$DATASET_HDF5" "$DATASET_ZARR" --overwrite || exit 1
fi
echo "Done."
