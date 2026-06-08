#!/bin/bash
# =============================================================================
# OpenBPMD SLURM array job script — one task per ligand subdirectory
#
# Do NOT submit this script directly. Use submit_array.sh instead, which
# counts ligand directories and sets --array automatically:
#
#   bash /path/to/OpenBPMD/slurm/submit_array.sh [pocket_dir]
#
# Each task enters one ligand subdirectory and writes output there.
# Logs are written to logs/openbpmd_<arrayid>_<taskid>.{log,err} inside
# the pocket directory (created by submit_array.sh before submission).
# =============================================================================

#SBATCH --job-name=openbpmd_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1            # pin GPU type with e.g. gpu:l4:1 or gpu:a10g:1
#SBATCH --mem=16G               # increase for systems larger than ~50k atoms
#SBATCH --time=16:00:00         # L4: GCMC ~1 h + 10x10 ns serial ~12 h
#SBATCH --output=logs/openbpmd_%A_%a.log
#SBATCH --error=logs/openbpmd_%A_%a.err

# =============================================================================
# USER CONFIGURATION — edit these paths before first use
# =============================================================================
OPENBPMD_SCRIPT=/path/to/OpenBPMD/openbpmd.py
GRAND_PATH=/path/to/grand       # source directory of the patched grand library
CONDA_BASE=/path/to/miniconda3  # base conda install (parent of envs/)
CONDA_ENV=openbpmd

# Job parameters
OUTPUT_DIR=openbpmd_results
LIG_RESNAME=MOL
NREPS=10
HILL_HEIGHT=0.3
STRUCTURE=solvated.rst7
PARAMETERS=solvated.prm7
# EXTRA_FLAGS="--no-grand"      # uncomment to skip GCMC water equilibration
# =============================================================================

# Activate conda environment
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Make the patched grand library importable
export PYTHONPATH="${GRAND_PATH}:${PYTHONPATH}"

# Build sorted list of ligand subdirectories that contain the expected input files.
# SLURM_SUBMIT_DIR is the pocket directory (set via --chdir in submit_array.sh).
mapfile -t LIGAND_DIRS < <(
    find "${SLURM_SUBMIT_DIR}" -mindepth 1 -maxdepth 1 -type d \
    | sort \
    | while IFS= read -r d; do
        [[ -f "${d}/${PARAMETERS}" && -f "${d}/${STRUCTURE}" ]] && echo "$d"
      done
)

if [[ ${SLURM_ARRAY_TASK_ID} -ge ${#LIGAND_DIRS[@]} ]]; then
    echo "ERROR: task ID ${SLURM_ARRAY_TASK_ID} exceeds ligand count ${#LIGAND_DIRS[@]}" >&2
    exit 1
fi

LIGAND_DIR="${LIGAND_DIRS[$SLURM_ARRAY_TASK_ID]}"

echo "======================================================"
echo "Array job     : ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "Node          : $(hostname)"
echo "GPU           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Ligand dir    : ${LIGAND_DIR}"
echo "Output dir    : ${LIGAND_DIR}/${OUTPUT_DIR}"
echo "======================================================"

cd "${LIGAND_DIR}"

python "${OPENBPMD_SCRIPT}" \
    -s "${STRUCTURE}" \
    -p "${PARAMETERS}" \
    -o "${OUTPUT_DIR}" \
    -lig_resname "${LIG_RESNAME}" \
    -nreps "${NREPS}" \
    -hill_height "${HILL_HEIGHT}" \
    ${EXTRA_FLAGS}

echo "======================================================"
echo "Task completed at $(date)"
echo "======================================================"
