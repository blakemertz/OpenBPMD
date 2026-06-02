# OpenBPMD — Session Context for Claude Code

This file captures the full context of the initial review session so work can continue
on a new workstation. Read this before doing anything else.

---

## What This Repository Is

OpenBPMD is an open-source Python reimplementation of Binding Pose Metadynamics (BPMD),
described in:

> Lukauskis et al., *J. Chem. Inf. Model.* 2022, 62, 6209–6216
> DOI: 10.1021/acs.jcim.2c01142

The paper PDF is at `Gervasio-open-binding-pose-metaD-JCIM2022.pdf` in the repo root.

**What it does:** Takes a solvated, parameterised protein–ligand complex (Amber .prm7/.rst7
or GROMACS .top/.gro), runs 10 × 10 ns metadynamics simulations biasing the ligand RMSD,
and scores pose stability using a composite of RMSD (PoseScore) and contact persistence
(ContactScore). Higher-scoring poses are more stable and more likely to be the true binding
mode. Intended as a post-docking pose-ranking filter.

**Scoring formula:**
```
CompScore = PoseScore − 5 × ContactScore
```
Lower CompScore = more stable pose.

**Validated performance:** 88% success rate at identifying the native pose within 2 Å RMSD
(on the Clark et al. 2016 benchmark set), but **only when grand water equilibration is used**
(see below). Without grand the baseline is ~69–71%.

**Important:** The README states that work has moved to `https://github.com/dlukauskis/OpenBPMD2`,
but that repo is not currently public. This repo is unmaintained upstream.

---

## Repository Layout

```
openbpmd.py              # CLI entry point; orchestrates the full pipeline
openbpmd/
  simulation.py          # minimize(), equilibrate(), produce() — OpenMM simulation code
  analysis.py            # get_pose_score(), get_contact_score(), collect_results(), plots
  __init__.py
  tests/
    test_openbpmd.py     # pytest suite
    files/               # test fixtures (pdb, dcd, prm7, rst7, npy, csv)
examples/
  input/                 # stable.prm7/.rst7, unstable.prm7/.rst7 (CDK2 system)
  stable_pose/rep_0..9/  # pre-computed example results
  unstable_pose/rep_0..9/
  example_analysis.ipynb
  example_analysis.py
  analysis_openbpmd.py
conda-recipe/meta.yaml
```

---

## Pipeline (as currently implemented)

```
openbpmd.py::main()
  1. minimize()       — energy minimisation to 10 kJ/mol tolerance (no dynamics)
  2. equilibrate()    — 500 ps NVT, 2 fs timestep, position restraints on solute heavy atoms
                        (5 kcal/mol/Å² harmonic constant via dummy-atom trick)
  3. [image_molecules via MDTraj → centred_equil_system.pdb]
  4. produce() × N    — N × 10 ns metadynamics, 4 fs timestep (HMR), NVT Langevin 300K
                        CV: ligand heavy-atom RMSD; Gaussian height configurable (default 0.3 kcal/mol);
                        width 0.002 nm; deposited every 1 ps; bias factor 4
                        Writes: trj.dcd, COLVAR.npy, sim_log.csv per rep
  5. collect_results() — averages last 2 ns of all reps → results.csv
```

Key simulation parameters (from paper):
- Equilibration: NVT, 2 fs timestep, 500 ps
- Production: NVT Langevin 300K, friction 1 ps⁻¹, 4 fs timestep (HMR: H mass = 4 Da)
- PME electrostatics, 10 Å cutoff
- 10 independent replicas (serial by default)
- Recommended hill height: 0.05 kcal/mol (gentler, more discriminating);
  0.3 kcal/mol also works but can be too aggressive for similar-energy poses

---

## Fixes Applied in This Session (already committed)

Commit: *"Fix compatibility with OpenMM 8.x and MDAnalysis 2.x"*

### simulation.py

1. **Removed `simtk.*` import fallback.** `simtk.openmm` was removed in OpenMM 8.0 (2022).
   Direct `from openmm import *` etc. is now used unconditionally.

2. **Added `_get_platform()` helper** (lines 14–27). Probes CUDA → OpenCL → CPU and returns
   the platform with appropriate precision properties (`CudaPrecision: mixed` for CUDA,
   `OpenCLPrecision: mixed` for OpenCL, empty dict for CPU). All three simulation functions
   now call `platform, properties = _get_platform()` instead of hardcoding CUDA.

3. **Fixed O(n²) COLVAR accumulation loop.** Replaced `np.append` growing the array on
   every iteration with a pre-allocated `np.zeros((n_iters+1, len(initial_cvs)))` array
   filled in-place. Saves ~5,000 array copies per 10 ns replica.

### analysis.py

4. **Fixed `r.rmsd` → `r.results.rmsd`** in `get_pose_score()`. MDAnalysis 2.0 (Nov 2021)
   moved all analysis results to `r.results.*`. The old attribute raises `AttributeError`
   on any current MDAnalysis install — this was a hard crash.

5. **Made `plot_all_reps()` frame-count dynamic.** Replaced hardcoded shape `(n_reps, 99)`
   and all `range(0, 99)` loops with `n_frames` derived from the first CSV file. Time axis
   now uses `np.linspace(0, n_frames * 0.1, n_frames)` (100 ps/frame assumed). Allows
   simulations of any length, not only 10 ns.

---

## The Grand Integration Gap — Primary Next Task

### The problem

**Grand (`essex-lab/grand`) is not integrated into the codebase at all.** The only mention
is a recommendation in README.md. This is a critical gap because:

- The paper's 88% success rate was achieved *with* grand GCMC water equilibration
- Without grand, the method achieves only ~69–71% (paper Figure 5)
- The improvement comes from correctly placing crystallographic bridging waters in the
  binding pocket before metadynamics — standard solvation (`gmx solvate`) misses them
  because buried water binding kinetics are too slow for short equilibration

### What grand does

Grand performs Grand Canonical Monte Carlo (GCMC) sampling of water molecules within a
sphere centred on the ligand binding site. Waters can be inserted/deleted during the
simulation, allowing the correct solvation shell to be found without requiring long
diffusion timescales.

**Grand repository:** `https://github.com/essex-lab/grand`
**Grand docs:** `https://grand.readthedocs.io`
**Citation:** Samways et al., *J. Chem. Inf. Model.* 2020, 60, 4436–4441

### The three-stage GCMC/MD protocol (from the paper)

This replaces/extends the current `equilibrate()` step, or runs as a new
`grand_equilibrate()` stage between `equilibrate()` and the first `produce()` call.

```
Stage 1a: 10,000 GCMC moves (pure MC, no MD)
Stage 1b: 1 ps interleaved GCMC/MD
           → 100 iterations of: [5 MD steps × 2 fs] + [1,000 GCMC moves]
Stage 2:  500 ps NPT MD (MonteCarloBarostat) to re-equilibrate box volume
Stage 3:  100,000 GCMC moves over 500 ps at the new volume
           → interleave GCMC moves with MD steps
```

### Grand API (current version, from readthedocs)

**`grand.utils` functions needed:**

```python
grand.utils.add_ghosts(topology, positions, ff='tip3p', n=10, pdb='gcmc-extra-wats.pdb')
# Adds n "ghost" (λ=0, fully decoupled) water molecules to topology/positions.
# MUST be called before createSystem() — it changes particle count.
# Returns modified (topology, positions).

grand.utils.remove_ghosts(topology, positions, ghosts=None, pdb='gcmc-removed-ghosts.pdb')
# Strips ghost waters from topology/positions after equilibration.
# ghosts = list of residue IDs with λ=0 status.
```

**Key sampler class for binding site work:**

```python
from grand.samplers import StandardGCMCSphereSampler

sampler = StandardGCMCSphereSampler(
    system,                # openmm.System (after add_ghosts and createSystem)
    topology,              # openmm.app.Topology
    temperature,           # e.g. 300*kelvin
    referenceAtoms,        # list of dicts: [{'name': 'C1', 'resname': 'MOL', 'resid': N}]
                           # defines the centre of the GCMC sphere (ligand atom)
    sphereRadius,          # e.g. 4.0*angstroms — should cover binding site
    excessChemicalPotential=-6.09*kilocalories_per_mole,  # TIP3P standard value
    standardVolume=30.345*angstroms**3,
    ghostFile='gcmc-ghost-wats.txt',
    log='gcmc.log',
    overwrite=True
)

# Initialise (must be called before move())
sampler.initialise(simulation.context, ghostResids=ghost_resids)

# Run n GCMC moves (insertion/deletion attempts)
sampler.move(simulation.context, n=1000)

# Get residue IDs of waters currently in λ=0 (ghost) state
ghost_resids = sampler.getWaterStatusResids(0)

# Report statistics to log file
sampler.report(simulation)
```

`GCMCSphereSampler` (parent of `StandardGCMCSphereSampler`) is the sphere-constrained
variant — correct for binding site work. `GCMCSystemSampler` covers the whole box
and is not appropriate here.

### Implementation plan for `grand_equilibrate()`

A new function `grand_equilibrate(eq_pdb, parm_file, structure_file, lig_resname, out_dir)`
needs to be added to `simulation.py`. It should:

1. **Add ghost waters** via `grand.utils.add_ghosts()` *before* `createSystem()`.
   Store the ghost residue IDs for later removal.

2. **Identify a `referenceAtoms` entry** for the GCMC sphere centre. The sphere should
   be centred on a ligand heavy atom (or ligand COM). The ligand residue ID must be
   extracted from the topology (currently the code uses only `lig_resname`, not `resid`
   — a small lookup step is needed).

3. **Create the OpenMM system** from the ghost-augmented topology.

4. **Instantiate `StandardGCMCSphereSampler`** with `sphereRadius` of ~4 Å.

5. **Run the three-stage protocol** using the NVT integrator for stages 1 and 3,
   adding a `MonteCarloBarostat` for stage 2 (NPT) and removing it before stage 3.

6. **Remove ghost waters** via `grand.utils.remove_ghosts()` and write the output PDB.
   This PDB becomes the new `eq_pdb` passed to all `produce()` calls.

### Critical architectural constraint

**Ghost waters change the topology particle count.** This means:

- `add_ghosts()` must be called before `AmberPrmtopFile.createSystem()`
- The same ghost-augmented topology must be used throughout grand equilibration
- After `remove_ghosts()`, a clean PDB is written; `produce()` can then use it with
  the *original* `parm_file` (no ghosts), since ghosts are stripped before production

The current code re-reads `parm_file` independently in each function (`minimize`,
`equilibrate`, `produce`). This is fine because grand equilibration is a self-contained
step that starts from the NVT-equilibrated PDB and ends by writing a new clean PDB —
it does not need to share topology objects with the other functions.

### Known grand compatibility concern

Grand's documentation still shows `simtk.openmm` type annotations, suggesting it may
not yet have been updated for OpenMM 8.0. **Check this first on the target workstation:**

```python
import grand  # if this fails with ImportError about simtk, grand needs updating
```

If grand has the same `simtk` problem, either install a newer version of grand or apply
the same fix (direct `from openmm import *`) to grand's source.

---

## Environment Setup

### Recommended conda environment (from README)

```bash
conda create -n openbpmd -c conda-forge openmm mdanalysis mdtraj parmed pytest cudatoolkit=11.8
conda activate openbpmd
pip install grand  # or: conda install -c essexlab grand
```

### Grand-specific dependencies (from grand docs)

Grand additionally requires: `openmmtools`, `lxml`. These come in via the grand install.

### Verify the install

```python
import openmm; print(openmm.__version__)   # should be 8.x
import MDAnalysis; print(MDAnalysis.__version__)  # should be 2.x
import grand; print(grand.__version__)
```

---

## Simulation Performance Reference

From the paper (NVIDIA GTX 2080Ti, ~44,000-atom CDK2 system):
- Throughput: **430 ns/day** (including Python/metadynamics overhead)
- 10 serial replicas × 10 ns: **~5.5 hours**
- 10 parallel replicas: **~30 minutes**

Estimates for other hardware at the same system size:

| GPU | Expected throughput | 10 serial reps | 10 parallel reps |
|---|---|---|---|
| GTX 2080Ti (reference) | 430 ns/day | ~5.5 h | ~30 min |
| A4000 | ~320–380 ns/day | ~6–8 h | ~35–45 min |
| L4 | ~220–290 ns/day | ~8–12 h | ~45–65 min |
| L40S | ~600–750 ns/day | ~3–4 h | ~18–25 min |

The A4000 and L4 are bandwidth-limited relative to the 2080Ti despite higher TFLOPS.
The L40S is the fastest of the three for MD due to its 864 GB/s bandwidth.

The production loop synchronises CPU↔GPU every 500 steps (2 ps) via
`getCollectiveVariables()`, which reduces GPU utilisation below what pure MD would give.
These estimates already account for that overhead.

---

## Known Remaining Issues (not yet fixed)

- **`collect_results()` in `analysis.py`** looks for `bpmd_results.csv` in each `rep_*`
  directory. The file is written as `bpmd_results.csv` in `openbpmd.py`. These match —
  no bug, but worth confirming the filename is consistent if the code is extended.

- **`produce()` is hardcoded to run serially.** The `openbpmd.py` main loop calls
  `produce()` in a `for idx in range(nreps)` loop. To run replicas in parallel (30 min
  instead of 5.5 h), replicas must be launched as separate processes (e.g., via
  `multiprocessing` or by running the script N times with different `-o` output dirs
  and then collecting results manually).

- **No CPU/OpenCL fallback for `grand` itself.** The `_get_platform()` fix in
  `simulation.py` handles OpenBPMD's own simulations. Grand creates its own OpenMM
  `Simulation` objects internally — whether it respects a user-specified platform
  or also hardcodes CUDA is not yet verified.

- **`test_openbpmd.py` imports are broken.** The test file does
  `import simulation; import analysis` (bare imports, not `from openbpmd import ...`).
  This works only if tests are run from within the `openbpmd/tests/` directory.
  The `sys.path.append` workaround at the top is fragile; tests should use the
  installed package.

---

## File State at End of Session

All changes above are committed. The only untracked file is the paper PDF:
`Gervasio-open-binding-pose-metaD-JCIM2022.pdf` (deliberately not committed — binary).

`git log --oneline -3` should show:
```
<hash>  Fix compatibility with OpenMM 8.x and MDAnalysis 2.x
9c727f9 Update README.md
e2664fe fixed conflicts
```
