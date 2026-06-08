#!/bin/bash
# =============================================================================
# OpenBPMD single-ligand SLURM job script
#
# Submit from the protein-ligand pair directory:
#   sbatch /path/to/OpenBPMD/slurm/openbpmd_single.slurm
#
# All output files are written to $OUTPUT_DIR inside the submission directory.
# =============================================================================

#SBATCH --job-name=openbpmd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1            # pin GPU type with e.g. gpu:l4:1 or gpu:a10g:1
#SBATCH --mem=16G               # increase for systems larger than ~50k atoms
#SBATCH --time=16:00:00         # L4: GCMC ~1 h + 10x10 ns serial ~12 h
#SBATCH --output=openbpmd_%j.log
#SBATCH --error=openbpmd_%j.err

# =============================================================================
# USER CONFIGURATION — edit these paths before first use
# =============================================================================
OPENBPMD_SCRIPT=/path/to/OpenBPMD/openbpmd.py
GRAND_PATH=/path/to/grand       # source directory of the patched grand library
CONDA_BASE=/path/to/miniconda3  # base conda install (parent of envs/)
CONDA_ENV=openbpmd

# Job parameters — override on the sbatch command line with --export if needed
OUTPUT_DIR=openbpmd_results
LIG_RESNAME=MOL
NREPS=10
HILL_HEIGHT=0.3
STRUCTURE=solvated.rst7
PARAMETERS=solvated.prm7
# EXTRA_FLAGS="--no-grand"      # uncomment to skip GCMC water equilibration
# =============================================================================

# Activate conda environment (works in non-interactive SLURM shells)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Make the patched grand library importable
export PYTHONPATH="${GRAND_PATH}:${PYTHONPATH}"

echo "======================================================"
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Node          : $(hostname)"
echo "GPU           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Ligand dir    : ${SLURM_SUBMIT_DIR}"
echo "Output dir    : ${SLURM_SUBMIT_DIR}/${OUTPUT_DIR}"
echo "======================================================"

cd "${SLURM_SUBMIT_DIR}"

python "${OPENBPMD_SCRIPT}" \
    -s "${STRUCTURE}" \
    -p "${PARAMETERS}" \
    -o "${OUTPUT_DIR}" \
    -lig_resname "${LIG_RESNAME}" \
    -nreps "${NREPS}" \
    -hill_height "${HILL_HEIGHT}" \
    ${EXTRA_FLAGS}

echo "======================================================"
echo "Job completed at $(date)"
echo "======================================================"
