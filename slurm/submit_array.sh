#!/usr/bin/env bash
# =============================================================================
# submit_array.sh — submit an OpenBPMD array job for all ligands in a pocket
#
# Usage:
#   bash /path/to/OpenBPMD/slurm/submit_array.sh [pocket_dir]
#
# If pocket_dir is omitted, the current working directory is used.
#
# The script discovers all immediate subdirectories of pocket_dir that contain
# both STRUCTURE and PARAMETERS input files, creates a logs/ directory, and
# submits openbpmd_array.slurm as a SLURM array job sized to match.
#
# Override default filenames via environment variables before calling:
#   STRUCTURE=my_complex.rst7 PARAMETERS=my_complex.prm7 bash submit_array.sh
# =============================================================================

set -euo pipefail

POCKET_DIR="${1:-$PWD}"
POCKET_DIR="$(realpath "${POCKET_DIR}")"
SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STRUCTURE="${STRUCTURE:-solvated.rst7}"
PARAMETERS="${PARAMETERS:-solvated.prm7}"

# Discover valid ligand subdirectories (sorted for reproducible task IDs)
DIRS=()
while IFS= read -r d; do
    [[ -f "${d}/${PARAMETERS}" && -f "${d}/${STRUCTURE}" ]] && DIRS+=("${d}")
done < <(find "${POCKET_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)

N="${#DIRS[@]}"
if [[ $N -eq 0 ]]; then
    echo "ERROR: no subdirectories containing ${STRUCTURE} + ${PARAMETERS} found in:" >&2
    echo "       ${POCKET_DIR}" >&2
    exit 1
fi

echo "Found ${N} ligand director$([ $N -eq 1 ] && echo y || echo ies):"
for d in "${DIRS[@]}"; do
    echo "  $(basename "${d}")"
done

# Create logs directory inside the pocket directory
mkdir -p "${POCKET_DIR}/logs"

echo ""
echo "Submitting SLURM array job (tasks 0–$((N-1))) from:"
echo "  ${POCKET_DIR}"

sbatch \
    --array="0-$((N-1))" \
    --chdir="${POCKET_DIR}" \
    "${SLURM_DIR}/openbpmd_array.slurm"
