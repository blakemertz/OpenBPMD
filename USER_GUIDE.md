# OpenBPMD User Guide

## Table of Contents

1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [Preparing Input Files](#preparing-input-files)
   - [Why Pre-Parameterized Input Is Required](#why-pre-parameterized-input-is-required)
   - [Step 1 — Ligand Parameterization with antechamber](#step-1--ligand-parameterization-with-antechamber)
   - [Step 2 — Building the Solvated Complex with tleap](#step-2--building-the-solvated-complex-with-tleap)
   - [Verifying Your Input Files](#verifying-your-input-files)
4. [Running OpenBPMD](#running-openbpmd)
   - [Command-Line Usage](#command-line-usage)
   - [All Options](#all-options)
5. [Pipeline Walkthrough](#pipeline-walkthrough)
   - [Stage 1 — Energy Minimization](#stage-1--energy-minimization)
   - [Stage 2 — NVT Equilibration](#stage-2--nvt-equilibration)
   - [Stage 3 — GCMC Water Equilibration](#stage-3--gcmc-water-equilibration)
   - [Stage 4 — Metadynamics Production Replicas](#stage-4--metadynamics-production-replicas)
   - [Stage 5 — Scoring](#stage-5--scoring)
6. [Complete Example: CDK2 System](#complete-example-cdk2-system)
7. [Output Files](#output-files)
8. [Interpreting Results](#interpreting-results)
9. [Performance Notes](#performance-notes)
10. [Troubleshooting](#troubleshooting)

---

## Overview

OpenBPMD is an open-source implementation of Binding Pose Metadynamics (BPMD), described in:

> Lukauskis et al., *J. Chem. Inf. Model.* 2022, 62, 6209–6216  
> DOI: 10.1021/acs.jcim.2c01142

It takes a solvated, parameterized protein–ligand complex in Amber (`.prm7`/`.rst7`) or GROMACS
(`.top`/`.gro`) format and outputs a stability score for the ligand pose. Lower scores indicate
more stable (more likely native) binding poses.

**Scoring formula:**

```
CompScore = PoseScore − 5 × ContactScore
```

- **PoseScore** — mean ligand RMSD over the last 2 ns of each replica
- **ContactScore** — fraction of native protein–ligand contacts preserved over the last 2 ns
- **CompScore** — composite score; lower is more stable

---

## Dependencies

- Python ≥ 3.9
- OpenMM ≥ 8.0
- MDAnalysis ≥ 2.0
- MDTraj
- ParmEd (parmed)
- grand ≥ 1.1 (GCMC water equilibration)
- NumPy, pandas

AmberTools is required for system preparation (see below) but is not a Python dependency —
it runs as a separate command-line toolchain.

---

## Preparing Input Files

### Why Pre-Parameterized Input Is Required

OpenBPMD runs molecular dynamics via OpenMM, which requires a fully parameterized description
of every atom in the system — masses, charges, Lennard-Jones parameters, and bonded terms. The
protein can be handled by a standard force field (e.g., Amber ff14SB), but **small-molecule
ligands are not covered by any standard force field** and must be parameterized individually
before running OpenBPMD.

The output of this preparation step is a pair of Amber files:

- `solvated.prm7` — topology and force field parameters for the whole system
- `solvated.rst7` — atomic coordinates and box vectors

These are the files passed to OpenBPMD with `-p` and `-s`.

---

### Step 1 — Ligand Parameterization with antechamber

`antechamber` assigns GAFF2 atom types and AM1-BCC partial charges to the ligand. You need:

- A ligand structure file with explicit bond orders (SDF or mol2 preferred — see note below)
- The ligand's net formal charge (integer)

```bash
# Preferred: convert SDF to mol2 with OpenBabel (preserves bond orders), then run antechamber
obabel -isdf ligand.sdf -omol2 -O ligand_ob.mol2

antechamber \
    -i  ligand_ob.mol2 \
    -fi mol2 \
    -o  ligand.mol2 \
    -fo mol2 \
    -c  bcc \
    -s  2 \
    -nc <net_charge>   # e.g. 0 for neutral, -1 for singly charged anion

# Generate missing bonded parameters not in GAFF2
parmchk2 \
    -i ligand.mol2 \
    -f mol2 \
    -o ligand.frcmod
```

**Key flags:**

| Flag | Meaning |
|------|---------|
| `-c bcc` | AM1-BCC charge method (fast, adequate for docking rescoring) |
| `-s 2` | Status level (print warnings) |
| `-nc` | Net formal charge of the ligand |

> **Why not PDB input?** PDB format does not store bond orders. `antechamber` must infer
> bond types from geometry alone, which fails for hypervalent atoms: sulfonyl (`S(=O)(=O)`),
> sulfoxide, phosphate, nitro groups, etc. Using SDF or mol2 input avoids this entirely.
> If your docking software exports SDF (Glide, Vina, GOLD), use SDF directly. If it exports
> PDB, convert to mol2 first via OpenBabel: `obabel -ipdb ligand.pdb -omol2 -O ligand_ob.mol2`.

If `antechamber` fails with charge-calculation errors, ensure the input structure has explicit
hydrogens at the correct protonation state. Use `reduce` or `OpenBabel` to add hydrogens
if needed.

---

### Step 2 — Building the Solvated Complex with tleap

`tleap` combines the protein, ligand, and water model into a single parameterized system. Save
the following as `tleap.in`:

```
source leaprc.protein.ff14SB       # Amber ff14SB for the protein
source leaprc.water.tip3p           # TIP3P water model
source leaprc.gaff2                 # GAFF2 for the ligand

# Load ligand parameters
loadamberparams ligand.frcmod
MOL = loadmol2 ligand.mol2

# Load the protein–ligand complex
# The PDB must have the ligand residue named MOL (matching the mol2 residue name)
complex = loadpdb protein_ligand.pdb

# Solvate in a truncated octahedron with 12 Å buffer
solvateoct complex TIP3PBOX 12.0

# Check for missing parameters before saving
check complex

# Write Amber parameter and coordinate files
saveamberparm complex solvated.prm7 solvated.rst7
quit
```

Run it:

```bash
tleap -f tleap.in
```

**Important:** the residue name in `ligand.mol2` and in `protein_ligand.pdb` must match and must
be the same string you pass to OpenBPMD with `-lig_resname` (default: `MOL`).

If `tleap` reports `unperturbed charge of the unit` warnings, the charge assignment in
antechamber may be incorrect — recheck `-nc`.

---

### Verifying Your Input Files

Before running OpenBPMD, confirm the files are usable:

```python
from openmm.app import AmberPrmtopFile, AmberInpcrdFile

parm = AmberPrmtopFile('solvated.prm7')
coords = AmberInpcrdFile('solvated.rst7')
system = parm.createSystem()   # should complete without error
print(f"System has {system.getNumParticles()} atoms")
```

Also verify the ligand residue name:

```bash
grep "^ATOM\|^HETATM" protein_ligand.pdb | awk '{print $4}' | sort -u
```

The ligand residue name shown must match the `-lig_resname` argument you will pass to OpenBPMD.

---

## Running OpenBPMD

### Command-Line Usage

```bash
python openbpmd.py \
    -s solvated.rst7 \
    -p solvated.prm7 \
    -o output_dir \
    -lig_resname MOL \
    -nreps 10 \
    -hill_height 0.05
```

### All Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-s`, `--structure` | str | `solvated.rst7` | Coordinate file (`.rst7` or `.gro`) |
| `-p`, `--parameters` | str | `solvated.prm7` | Topology/parameter file (`.prm7` or `.top`) |
| `-o`, `--output` | str | `.` | Output directory (created if absent) |
| `-lig_resname` | str | `MOL` | Residue name of the ligand |
| `-nreps` | int | `10` | Number of independent metadynamics replicas |
| `-hill_height` | float | `0.3` | Metadynamics hill height in kcal/mol |
| `--no-grand` | flag | off | Skip GCMC water equilibration (reduces accuracy) |

**Recommended hill height:** `0.05` kcal/mol gives more discriminating scores for similar poses.
`0.3` kcal/mol (the default) is faster but may be too aggressive for closely ranked poses.

---

## Pipeline Walkthrough

### Stage 1 — Energy Minimization

Minimizes the input structure to a gradient tolerance of 10 kJ/mol. Removes clashes
introduced during solvation without running dynamics.

Output: `{output_dir}/minimized_system.pdb`

This stage is skipped automatically if the file already exists.

---

### Stage 2 — NVT Equilibration

Runs 500 ps of NVT MD at 300 K with harmonic position restraints (5 kcal/mol/Å²) on all
solute heavy atoms (protein + ligand). Uses a 2 fs timestep. This relaxes the solvent while
keeping the complex close to its input geometry.

Output: `{output_dir}/equil_system.pdb`

This stage is skipped automatically if the file already exists.

---

### Stage 3 — GCMC Water Equilibration

Runs Grand Canonical Monte Carlo (GCMC) sampling of water molecules in the ligand binding site
using the `grand` library. This correctly places crystallographic bridging waters that are
kinetically inaccessible on short MD timescales.

**Why this matters:** The paper reports 88% pose-ranking accuracy *with* GCMC water equilibration
versus ~69–71% without it. Binding site waters that bridge protein–ligand interactions are
critical for scoring accuracy but are rarely placed correctly by standard solvation.

The protocol has three stages:

| Stage | Protocol | Purpose |
|-------|----------|---------|
| 1a | 10,000 pure GCMC moves | Seed the binding site with water |
| 1b | 100 × (1,000 GCMC moves + 5 MD steps) = 1 ps | Relax water positions |
| 2 | 500 ps NPT MD | Re-equilibrate box volume |
| 3 | 500 × (500 MD steps + 200 GCMC moves) = 500 ps | Converge water occupancies |

Output: `{output_dir}/grand_equil_system.pdb`, `gcmc.log`, `gcmc-ghost-wats.txt`

This stage is skipped if `grand_equil_system.pdb` already exists, or if `--no-grand` is passed.

---

### Stage 4 — Metadynamics Production Replicas

Runs `-nreps` independent 10 ns metadynamics simulations biasing the ligand RMSD. Uses:

- 4 fs timestep with hydrogen mass repartitioning (HMR, H mass = 4 Da)
- NVT Langevin thermostat at 300 K, friction 1 ps⁻¹
- PME electrostatics, 10 Å cutoff
- Gaussian hills of width 0.002 nm deposited every 1 ps, bias factor 4

Each replica runs serially. Output per replica in `{output_dir}/rep_{i}/`:

- `trj.dcd` — trajectory (1 frame per 100 ps)
- `COLVAR.npy` — time-resolved ligand RMSD
- `sim_log.csv` — temperature, speed, progress
- `bias_*.npy` — deposited bias grids

---

### Stage 5 — Scoring

After each replica, PoseScore and ContactScore are computed from the trajectory. The final
`results.csv` is written after all replicas complete.

---

## Complete Example: CDK2 System

This example uses the CDK2 benchmark system included in the repository.

```bash
# Navigate to the repository root
cd /path/to/OpenBPMD

# Run the stable pose example with 1 replica for a quick test
python openbpmd.py \
    -s examples/input/stable.rst7 \
    -p examples/input/stable.prm7 \
    -o test_output \
    -lig_resname MOL \
    -nreps 1 \
    -hill_height 0.05

# Expected output directory structure after completion:
# test_output/
#   minimized_system.pdb
#   equil_system.pdb
#   grand_equil_system.pdb       ← GCMC-equilibrated starting structure
#   centred_grand_equil_system.pdb
#   gcmc.log                     ← GCMC acceptance rates and N statistics
#   gcmc-ghost-wats.txt          ← ghost water residue IDs per frame
#   gcmc-extra-wats.pdb          ← ghost water PDB (visualization)
#   rep_0/
#     trj.dcd
#     COLVAR.npy
#     sim_log.csv
#     bpmd_results.csv
#   results.csv                  ← final scores
```

To skip GCMC (faster, lower accuracy):

```bash
python openbpmd.py \
    -s examples/input/stable.rst7 \
    -p examples/input/stable.prm7 \
    -o test_output_nogrand \
    -lig_resname MOL \
    -nreps 1 \
    --no-grand
```

To run both poses and compare:

```bash
python openbpmd.py -s examples/input/stable.rst7   -p examples/input/stable.prm7   -o stable_out   -lig_resname MOL -nreps 10 -hill_height 0.05
python openbpmd.py -s examples/input/unstable.rst7 -p examples/input/unstable.prm7 -o unstable_out -lig_resname MOL -nreps 10 -hill_height 0.05
```

The stable pose should yield a lower (more negative) CompScore than the unstable pose.

---

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `minimized_system.pdb` | `output_dir/` | Energy-minimized structure |
| `equil_system.pdb` | `output_dir/` | NVT-equilibrated structure |
| `grand_equil_system.pdb` | `output_dir/` | GCMC water-equilibrated structure |
| `centred_grand_equil_system.pdb` | `output_dir/` | PBC-imaged version for analysis |
| `gcmc.log` | `output_dir/` | GCMC acceptance rates and N(water) per step |
| `gcmc-ghost-wats.txt` | `output_dir/` | Ghost water residue IDs per GCMC frame |
| `gcmc-extra-wats.pdb` | `output_dir/` | Ghost-augmented topology PDB |
| `trj.dcd` | `output_dir/rep_N/` | Metadynamics trajectory |
| `COLVAR.npy` | `output_dir/rep_N/` | Ligand RMSD time series (nm) |
| `sim_log.csv` | `output_dir/rep_N/` | Simulation log (temperature, speed) |
| `bpmd_results.csv` | `output_dir/rep_N/` | Per-frame PoseScore, ContactScore, CompScore |
| `results.csv` | `output_dir/` | Final averaged scores across all replicas |

---

## Interpreting Results

`results.csv` contains one row per replica with columns: `CompScore`, `PoseScore`, `ContactScore`.

```
CompScore = PoseScore − 5 × ContactScore
```

- **Lower CompScore → more stable pose → more likely native binding mode**
- A CompScore difference of ≥ 1–2 kcal/mol between poses is typically significant
- The paper validates a success rate of 88% (pose within 2 Å of crystal structure ranked first)
  on the Clark et al. 2016 benchmark set when GCMC is used

**Rule of thumb from the paper:**

| CompScore | Interpretation |
|-----------|---------------|
| < −2 | Very stable pose — likely correct |
| −2 to 0 | Marginally stable |
| > 0 | Unstable pose — likely incorrect |

These cutoffs are system-dependent. Always compare poses relative to each other rather than
using absolute thresholds.

**Checking GCMC quality:** Inspect `gcmc.log` to verify water exchange occurred:

```
# Good: non-zero acceptance rate and varying N
100 move(s) completed (42 accepted (42.0000 %)). Current N = 4. Average N = 3.8
```

If acceptance rate is 0% throughout, the sphere radius may be too small or the ligand
residue name is incorrect.

---

## Performance Notes

Approximate throughput on the ~44,000-atom CDK2 system (from the paper):

| Hardware | Throughput | 10 serial replicas |
|----------|-----------|-------------------|
| RTX 2080Ti (reference) | 430 ns/day | ~5.5 h |
| A4000 | ~350 ns/day | ~7 h |
| L4 | ~250 ns/day | ~10 h |
| L40S | ~700 ns/day | ~3.5 h |

GCMC equilibration adds approximately 30–60 minutes depending on GPU speed.

The production loop is serial by default. To run replicas in parallel, launch multiple
`openbpmd.py` invocations with different output directories, then collect results manually.

---

## Troubleshooting

**`antechamber: Fatal Error! Cannot properly run bondtype` (sulfur or other hypervalent atom)**  
`antechamber` failed to assign bond types because the input PDB has no bond-order information.
This is common for sulfonyl (`S(=O)(=O)`), sulfoxide, phosphate, and nitro groups. Switch to
mol2 or SDF input:
```bash
obabel -ipdb ligand.pdb -omol2 -O ligand_ob.mol2
antechamber -i ligand_ob.mol2 -fi mol2 -o ligand.mol2 -fo mol2 -c bcc -s 2 -nc <charge>
```

**`ValueError: No template found for residue MOL`**  
The force field cannot find parameters for the ligand. Ensure the `.prm7` file was generated
by tleap with the ligand's `.frcmod` loaded. Do not mix Amber input with `--no-grand` and
a system built from GROMACS topology files.

**`ValueError: Ligand residue 'MOL' not found in topology`**  
The `-lig_resname` argument does not match the residue name in your parameter file. Check
the residue name with:  
`grep -m5 "^ATOM\|^HETATM" equil_system.pdb`

**GCMC acceptance rate is 0%**  
The GCMC sphere (default 4 Å) may not overlap any water-accessible space around the ligand.
This can happen if the ligand is fully buried or if the binding site has no bridging waters.
Consider running with `--no-grand` for such systems.

**Simulation crashes with `NaN` energy**  
The input structure has bad geometry (clashes or incorrect chirality). Re-run the
AmberTools preparation, ensure the ligand PDB has correct stereochemistry, and verify
the minimization step completed without error.

**`ImportError: No module named 'grand'`**  
Install grand: `pip install grand` or `conda install -c essexlab grand`.
Ensure the installed version is compatible with OpenMM 8.x (see the grand repository
for version requirements).
