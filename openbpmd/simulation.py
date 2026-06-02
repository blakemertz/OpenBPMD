# OpenMM
from openmm import *
from openmm.app import *
from openmm.unit import *
from openmm.app.metadynamics import *

# The rest
import numpy as np
import mdtraj as md
import MDAnalysis as mda
import grand
from grand.samplers import StandardGCMCSphereSampler
import os


def _get_platform():
    """Return the fastest available OpenMM platform and its precision properties."""
    for name in ('CUDA', 'OpenCL', 'CPU'):
        try:
            platform = Platform.getPlatformByName(name)
            if name == 'CUDA':
                return platform, {'CudaPrecision': 'mixed'}
            elif name == 'OpenCL':
                return platform, {'OpenCLPrecision': 'mixed'}
            else:
                return platform, {}
        except Exception:
            continue
    raise RuntimeError("No OpenMM platform available.")


# TODO: add default types
def minimize(
    parm_file, structure_file, out_dir, min_file_name
):
    """An energy minimization function down with an energy tolerance
    of 10 kJ/mol.

    Parameters
    ----------
    parm_file : str, path to the parameter/topology file
        Used to create the OpenMM System object.
    structure_file : str, path to structure/coordinate file
        3D coordinates of atoms used to create an OpenMM system.
    out_dir : str
        Directory to write the outputs.
    min_file_name : str
        Name of the minimized PDB file to write.
    """
    
    if structure_file.endswith('.gro'):
        coords = GromacsGroFile(structure_file)
        box_vectors = coords.getPeriodicBoxVectors()
        parm = GromacsTopFile(parm_file, periodicBoxVectors=box_vectors)
    else:
        coords = AmberInpcrdFile(structure_file)
        parm = AmberPrmtopFile(parm_file)
        
    system = parm.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=1*nanometers,
        constraints=HBonds,
    )

    # Select fastest available platform
    platform, properties = _get_platform()

    # Set up the simulation parameters
    # Langevin integrator at 300 K w/ 1 ps^-1 friction coefficient
    # and a 2-fs timestep
    # NOTE - no dynamics performed, but required for setting up
    # the OpenMM system.
    integrator = LangevinIntegrator(300*kelvin, 1/picosecond,
                                    0.002*picoseconds)
    simulation = Simulation(parm.topology, system, integrator, platform,
                            properties)
    simulation.context.setPositions(coords.positions)

    # Minimize the system - no predefined number of steps
    simulation.minimizeEnergy()

    # Write out the minimized system to use w/ MDAnalysis
    positions = simulation.context.getState(getPositions=True).getPositions()
    out_file = os.path.join(out_dir,min_file_name)
    PDBFile.writeFile(simulation.topology, positions,
                      open(out_file, 'w'))

    return None

# TODO: add default types
def equilibrate(
    min_pdb, parm_file, structure_file, out_dir, eq_file_name
):
    """A function that does a 500 ps NVT equilibration with position
    restraints, with a 5 kcal/mol/A**2 harmonic constant on solute heavy
    atoms, using a 2 fs timestep.

    Parameters
    ----------
    min_pdb : str
        Name of the minimized PDB file.
    parm_file : str
        The name of the parameter or topology file of the system.
    structure_file : str
        The name of the coordinate file of the system.
    out_dir : str
        Directory to write the outputs to.
    eq_file_name : str
        Name of the equilibrated PDB file to write.
    """
    if structure_file.endswith('.gro'):
        coords = GromacsGroFile(structure_file)
        box_vectors = coords.getPeriodicBoxVectors()
        parm = GromacsTopFile(parm_file, periodicBoxVectors=box_vectors)
    else:
        coords = AmberInpcrdFile(structure_file)
        parm = AmberPrmtopFile(parm_file)
    
    # Get the solute heavy atom indices to use
    # for defining position restraints during equilibration
    universe = mda.Universe(min_pdb,
                            format='XPDB', in_memory=True)
    solute_heavy_atom_idx = universe.select_atoms('not resname WAT and\
                                                   not resname SOL and\
                                                   not resname HOH and\
                                                   not resname CL and \
                                                   not resname NA and \
                                                   not name H*').indices
    # Necessary conversion to int from numpy.int64,
    # b/c it breaks OpenMM C++ function
    solute_heavy_atom_idx = [int(idx) for idx in solute_heavy_atom_idx]

    # Add the restraints.
    # We add a dummy atoms with no mass, which are therefore unaffected by
    # any kind of scaling done by barostat (if used). And the atoms are
    # harmonically restrained to the dummy atom. We have to redefine the
    # system, b/c we're adding new particles and this would clash with
    # modeller.topology.
    system = parm.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=1 * nanometers,
        constraints=HBonds,
    )
    # Add the harmonic restraints on the positions
    # of specified atoms
    restraint = HarmonicBondForce()
    restraint.setUsesPeriodicBoundaryConditions(True)
    system.addForce(restraint)
    nonbonded = [force for force in system.getForces()
                 if isinstance(force, NonbondedForce)][0]
    dummyIndex = []
    input_positions = PDBFile(min_pdb).getPositions()
    positions = input_positions
    # Go through the indices of all atoms that will be restrained
    for i in solute_heavy_atom_idx:
        j = system.addParticle(0)
        # ... and add a dummy/ghost atom next to it
        nonbonded.addParticle(0, 1, 0)
        # ... that won't interact with the restrained atom 
        nonbonded.addException(i, j, 0, 1, 0)
        # ... but will be have a harmonic restraint ('bond')
        # between the two atoms
        restraint.addBond(i, j, 0 * nanometers,
                          5*kilocalories_per_mole/angstrom**2)
        dummyIndex.append(j)
        input_positions.append(positions[i])

    integrator = LangevinIntegrator(
        300 * kelvin, 1 / picosecond, 0.002 * picoseconds
    )
    platform, properties = _get_platform()
    sim = Simulation(parm.topology, system, integrator,
                     platform, properties)
    sim.context.setPositions(input_positions)
    integrator.step(250000)  # run 500 ps of equilibration
    all_positions = sim.context.getState(
        getPositions=True, enforcePeriodicBox=True).getPositions()
    # we don't want to write the dummy atoms, so we only
    # write the positions of atoms up to the first dummy atom index
    relevant_positions = all_positions[:dummyIndex[0]]
    out_file = os.path.join(out_dir,eq_file_name)
    PDBFile.writeFile(sim.topology, relevant_positions,
                      open(out_file, 'w'))

    return None

# TODO: add default types
def produce(
    out_dir, idx, lig_resname, eq_pdb, parm_file, structure_file,
    set_hill_height, set_sim_time):
    """An OpenBPMD production simulation function. Ligand RMSD is biased with
    metadynamics. The integrator uses a 4 fs time step and
    runs for 10 ns, writing a frame every 100 ps.

    Writes a 'trj.dcd', 'COLVAR.npy', 'bias_*.npy' and 'sim_log.csv' files
    during the metadynamics simulation in the '{out_dir}/rep_{idx}' directory.
    After the simulation is done, it analyses the trajectories and writes a
    'bpm_results.csv' file with time-resolved PoseScore and ContactScore.

    Parameters
    ----------
    out_dir : str
        Directory where your equilibration PDBs and 'rep_*' dirs are at.
    idx : int
        Current replica index.
    lig_resname : str
        Residue name of the ligand.
    eq_pdb : str
        Name of the PDB for equilibrated system.
    parm_file : str
        The name of the parameter or topology file of the system.
    structure_file : str
        The name of the coordinate file of the system.
    set_hill_height : float
        Metadynamic hill height, in kcal/mol.
    set_sim_time : int
        Metadynamic simulation time, in ns.
    """
    if structure_file.endswith('.gro'):
        coords = GromacsGroFile(structure_file)
        box_vectors = coords.getPeriodicBoxVectors()
        parm = GromacsTopFile(parm_file, periodicBoxVectors=box_vectors)
    else:
        coords = AmberInpcrdFile(structure_file)
        parm = AmberPrmtopFile(parm_file)
        
    # First, assign the replica directory to which we'll write the files
    write_dir = os.path.join(out_dir,f'rep_{idx}')
    # Get the anchor atoms by ...
    universe = mda.Universe(eq_pdb,
                            format='XPDB', in_memory=True)
    # ... finding the protein's COM ...
    prot_com = universe.select_atoms('protein').center_of_mass()
    x, y, z = prot_com[0], prot_com[1], prot_com[2]
    # ... and taking the heavy backbone atoms within 5A of the COM
    sel_str = f'point {x} {y} {z} 5 and backbone and not name H*'
    anchor_atoms = universe.select_atoms(sel_str)
    # ... or 10 angstrom
    if len(anchor_atoms) == 0:
        sel_str = f'point {x} {y} {z} 10 and backbone and not name H*'
        anchor_atoms = universe.select_atoms(sel_str)

    anchor_atom_idx = anchor_atoms.indices.tolist()

    # Get indices of ligand heavy atoms
    lig = universe.select_atoms(f'resname {lig_resname} and not name H*')

    lig_ha_idx = lig.indices.tolist()

    # Set up the system to run metadynamics
    system = parm.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=1 * nanometers,
        constraints=HBonds,
        hydrogenMass=4*amu
    )
    # get the atom positions for the system from the equilibrated
    # system
    input_positions = PDBFile(eq_pdb).getPositions()

    # Add an 'empty' flat-bottom restraint to fix the issue with PBC.
    # Without one, RMSDForce object fails to account for PBC.
    k = 0*kilojoules_per_mole  # NOTE - 0 kJ/mol constant
    upper_wall = 10.00*nanometer
    fb_eq = '(k/2)*max(distance(g1,g2) - upper_wall, 0)^2'
    upper_wall_rest = CustomCentroidBondForce(2, fb_eq)
    upper_wall_rest.addGroup(lig_ha_idx)
    upper_wall_rest.addGroup(anchor_atom_idx)
    upper_wall_rest.addBond([0, 1])
    upper_wall_rest.addGlobalParameter('k', k)
    upper_wall_rest.addGlobalParameter('upper_wall', upper_wall)
    upper_wall_rest.setUsesPeriodicBoundaryConditions(True)
    system.addForce(upper_wall_rest)

    alignment_indices = lig_ha_idx + anchor_atom_idx

    rmsd = RMSDForce(input_positions, alignment_indices)
    # Set up the typical metadynamics parameters
    grid_min, grid_max = 0.0, 1.0  # nm
    hill_height = set_hill_height*kilocalories_per_mole
    hill_width = 0.002  # nm, also known as sigma

    grid_width = hill_width / 5
    # 'grid' here refers to the number of grid points
    grid = int(abs(grid_min - grid_max) / grid_width)

    rmsd_cv = BiasVariable(rmsd, grid_min, grid_max, hill_width,
                           False, gridWidth=grid)

    # define the metadynamics object
    # deposit bias every 1 ps, BF = 4, write bias every ns
    meta = Metadynamics(system, [rmsd_cv], 300.0*kelvin, 4.0, hill_height,
                        250, biasDir=write_dir,
                        saveFrequency=250000)

    # Set up and run metadynamics
    integrator = LangevinIntegrator(
        300 * kelvin, 1.0 / picosecond, 0.004 * picoseconds
    )
    platform, properties = _get_platform()

    simulation = Simulation(parm.topology, system, integrator, platform,
                            properties)
    simulation.context.setPositions(input_positions)

    trj_name = os.path.join(write_dir, 'trj.dcd')

    sim_time = set_sim_time  # ns
    steps = 250000 * sim_time

    simulation.reporters.append(DCDReporter(trj_name, 25000))  # every 100 ps
    simulation.reporters.append(StateDataReporter(
                                os.path.join(write_dir, 'sim_log.csv'), 250000,
                                step=True, temperature=True, progress=True,
                                remainingTime=True, speed=True,
                                totalSteps=steps, separator=','))  # every 1 ns

    n_iters = int(steps) // 500
    initial_cvs = meta.getCollectiveVariables(simulation)
    colvar_array = np.zeros((n_iters + 1, len(initial_cvs)))
    colvar_array[0] = initial_cvs
    for it, step_start in enumerate(range(0, int(steps), 500)):
        if step_start % 25000 == 0:
            # log the stored COLVAR every 100 ps
            np.save(os.path.join(write_dir, 'COLVAR.npy'), colvar_array[:it + 1])
        meta.step(simulation, 500)
        colvar_array[it + 1] = meta.getCollectiveVariables(simulation)
    np.save(os.path.join(write_dir, 'COLVAR.npy'), colvar_array)

    # center everything using MDTraj, to fix any PBC imaging issues
    # mdtraj can't use GMX TOP, so we have to specify the GRO file instead
    if structure_file.endswith('.gro'):
        mdtraj_top = structure_file
    else:
        mdtraj_top = parm_file
    mdu = md.load(trj_name, top=mdtraj_top)
    mdu.image_molecules()
    mdu.save(trj_name)

    return None


def grand_equilibrate(
    eq_pdb, parm_file, structure_file, lig_resname, out_dir, grand_eq_file_name
):
    """Run Grand Canonical Monte Carlo (GCMC) water equilibration using the
    three-stage protocol from Lukauskis et al. 2022. Correctly places
    crystallographic bridging waters in the binding pocket before production.

    Stages:
        1a. 10,000 pure GCMC moves (NVT)
        1b. 1 ps interleaved GCMC/MD (100 iterations of 1,000 moves + 5 steps)
        2.  500 ps NPT MD to re-equilibrate box volume
        3.  500 ps interleaved GCMC/MD at new volume (100,000 total GCMC moves)

    Writes 'grand_equil_system.pdb' (or grand_eq_file_name) to out_dir.

    Parameters
    ----------
    eq_pdb : str
        Path to the NVT-equilibrated PDB (output of equilibrate()).
    parm_file : str
        Parameter/topology file (.prm7 or .top).
    structure_file : str
        Coordinate file (.rst7 or .gro), used only to detect Amber vs GROMACS.
    lig_resname : str
        Residue name of the ligand.
    out_dir : str
        Directory to write output files.
    grand_eq_file_name : str
        Name of the output PDB file with GCMC-equilibrated waters.
    """
    # Load topology and coordinates
    if structure_file.endswith('.gro'):
        box_vectors = GromacsGroFile(structure_file).getPeriodicBoxVectors()
        parm = GromacsTopFile(parm_file, periodicBoxVectors=box_vectors)
    else:
        parm = AmberPrmtopFile(parm_file)

    coords = PDBFile(eq_pdb)

    # Create the OpenMM system from the ORIGINAL prmtop/top BEFORE adding ghosts.
    # AmberPrmtopFile/GromacsTopFile.createSystem() use their own internal
    # topology, which already includes all ligand parameters from the input
    # parameter file — no XML force field look-up required.
    system = parm.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=1*nanometers,
        constraints=HBonds,
    )

    # Augment the topology and positions with ghost TIP3P waters.
    # grand.utils.add_ghosts() appends HOH residues to the topology and places
    # them at random positions within the simulation box.
    topology, positions, ghost_resids = grand.utils.add_ghosts(
        parm.topology, coords.positions,
        ff='tip3p', n=15,
        pdb=os.path.join(out_dir, 'gcmc-extra-wats.pdb')
    )

    # Extend the OpenMM system to include the ghost water particles.
    # We read mass, charge, sigma, epsilon, and constraint distances directly
    # from an existing water in the system — no hardcoded force field values.
    nonbonded = None
    for i in range(system.getNumForces()):
        if isinstance(system.getForce(i), NonbondedForce):
            nonbonded = system.getForce(i)
            break

    # Extract TIP3P parameters and intra-water constraint distances from the
    # first water molecule already present in the parameterised system.
    wat_params = []       # [(mass, charge, sigma, epsilon), ...] one per atom
    wat_constraints = []  # [(local_idx_a, local_idx_b, distance), ...]
    for residue in parm.topology.residues():
        if residue.name in ('WAT', 'HOH', 'SOL'):
            wat_atoms = list(residue.atoms())
            atom_indices = {a.index for a in wat_atoms}
            local_map = {a.index: i for i, a in enumerate(wat_atoms)}
            for atom in wat_atoms:
                mass = system.getParticleMass(atom.index)
                charge, sigma, epsilon = nonbonded.getParticleParameters(atom.index)
                wat_params.append((mass, charge, sigma, epsilon))
            for k in range(system.getNumConstraints()):
                p1, p2, dist = system.getConstraintParameters(k)
                if p1 in atom_indices and p2 in atom_indices:
                    wat_constraints.append((local_map[p1], local_map[p2], dist))
            break

    if not wat_params:
        raise ValueError(
            "No water molecules (WAT/HOH/SOL) found in the parameter file. "
            "grand_equilibrate() requires an explicitly solvated system."
        )

    # Add one set of ghost particles per ghost residue
    for ghost_resid in ghost_resids:
        # Locate ghost atoms in the augmented topology
        ghost_atoms = []
        for resid, residue in enumerate(topology.residues()):
            if resid == ghost_resid:
                ghost_atoms = list(residue.atoms())
                break

        # Register particles and nonbonded parameters
        for i, atom in enumerate(ghost_atoms):
            mass, charge, sigma, epsilon = wat_params[i]
            system.addParticle(mass)
            nonbonded.addParticle(charge, sigma, epsilon)

        # Reproduce the same intra-water constraints as the template water
        for local_a, local_b, dist in wat_constraints:
            system.addConstraint(
                ghost_atoms[local_a].index,
                ghost_atoms[local_b].index,
                dist,
            )

        # Exclude intra-water nonbonded interactions (1-2 and 1-3 pairs)
        for i in range(len(ghost_atoms)):
            for j in range(i + 1, len(ghost_atoms)):
                nonbonded.addException(
                    ghost_atoms[i].index, ghost_atoms[j].index,
                    0 * elementary_charge**2,
                    1 * nanometers,
                    0 * kilojoules_per_mole,
                )

    # Find the first heavy atom of the ligand to define the GCMC sphere centre
    ref_atoms = []
    for residue in topology.residues():
        if residue.name == lig_resname:
            for atom in residue.atoms():
                if not atom.name.startswith('H'):
                    ref_atoms.append({
                        'name': atom.name,
                        'resname': lig_resname,
                        'resid': residue.id,
                    })
                    break
            break
    if not ref_atoms:
        raise ValueError(
            f"Ligand residue '{lig_resname}' not found in topology. "
            "Check -lig_resname."
        )

    # Instantiate the GCMC sampler
    gcmc_mover = StandardGCMCSphereSampler(
        system=system,
        topology=topology,
        temperature=300*kelvin,
        referenceAtoms=ref_atoms,
        sphereRadius=4.0*angstroms,
        excessChemicalPotential=-6.09*kilocalories_per_mole,
        standardVolume=30.345*angstroms**3,
        ghostFile=os.path.join(out_dir, 'gcmc-ghost-wats.txt'),
        log=os.path.join(out_dir, 'gcmc.log'),
        overwrite=True,
    )

    platform, properties = _get_platform()

    # ------------------------------------------------------------------ #
    # Stage 1: GCMC equilibration of binding site waters (NVT)            #
    # ------------------------------------------------------------------ #
    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
    simulation = Simulation(topology, system, integrator, platform, properties)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(300*kelvin)
    simulation.context.setPeriodicBoxVectors(*topology.getPeriodicBoxVectors())

    gcmc_mover.initialise(simulation.context, ghost_resids)
    # Clear any waters currently in the sphere to start fresh
    gcmc_mover.deleteWatersInGCMCSphere()

    # Stage 1a: 10,000 pure GCMC moves
    print("  GCMC stage 1a: 10,000 insertion/deletion moves...")
    gcmc_mover.move(simulation.context, 10000)

    # Minimise after pure GCMC to resolve any steric clashes introduced by
    # freshly inserted waters before MD steps begin.
    print("  Minimizing after stage 1a...")
    simulation.minimizeEnergy(maxIterations=500)
    simulation.context.setVelocitiesToTemperature(300*kelvin)

    # Stage 1b: 1 ps interleaved GCMC/MD (100 × [1,000 moves + 5 MD steps])
    # Use 1 fs timestep here for extra stability during frequent insertions.
    print("  GCMC stage 1b: 1 ps interleaved GCMC/MD...")
    integrator_1b = LangevinIntegrator(300*kelvin, 1/picosecond, 0.001*picoseconds)
    simulation_1b = Simulation(topology, system, integrator_1b, platform, properties)
    state_1a = simulation.context.getState(
        getPositions=True, getVelocities=True, enforcePeriodicBox=True
    )
    simulation_1b.context.setPositions(state_1a.getPositions())
    simulation_1b.context.setVelocities(state_1a.getVelocities())
    simulation_1b.context.setPeriodicBoxVectors(*state_1a.getPeriodicBoxVectors())
    gcmc_mover.context = simulation_1b.context
    for i in range(100):
        gcmc_mover.move(simulation_1b.context, 1000)
        gcmc_mover.report(simulation_1b)
        simulation_1b.step(10)  # 10 × 1 fs = 10 fs

    # ------------------------------------------------------------------ #
    # Stage 2: 500 ps NPT MD to re-equilibrate box volume                 #
    # ------------------------------------------------------------------ #
    print("  GCMC stage 2: 500 ps NPT equilibration...")
    system.addForce(MonteCarloBarostat(1*bar, 300*kelvin, 25))

    integrator2 = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
    simulation2 = Simulation(topology, system, integrator2, platform, properties)
    state = simulation_1b.context.getState(
        getPositions=True, getVelocities=True, enforcePeriodicBox=True
    )
    simulation2.context.setPositions(state.getPositions())
    simulation2.context.setVelocities(state.getVelocities())
    simulation2.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    simulation2.step(250000)  # 500 ps

    # ------------------------------------------------------------------ #
    # Stage 3: 100,000 GCMC moves over 500 ps at new box volume           #
    # ------------------------------------------------------------------ #
    print("  GCMC stage 3: 500 ps interleaved GCMC/MD at new volume...")

    # Remove the barostat before stage 3 (back to NVT)
    for i in range(system.getNumForces()):
        if isinstance(system.getForce(i), MonteCarloBarostat):
            system.removeForce(i)
            break

    integrator3 = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
    simulation3 = Simulation(topology, system, integrator3, platform, properties)
    state2 = simulation2.context.getState(
        getPositions=True, getVelocities=True, enforcePeriodicBox=True
    )
    simulation3.context.setPositions(state2.getPositions())
    simulation3.context.setVelocities(state2.getVelocities())
    simulation3.context.setPeriodicBoxVectors(*state2.getPeriodicBoxVectors())

    # Re-initialise sampler with current ghost list at new box dimensions
    gcmc_mover.initialise(simulation3.context, gcmc_mover.getWaterStatusResids(0))

    # 500 iterations × (500 MD steps + 200 GCMC moves) = 500 ps / 100k moves
    for i in range(500):
        simulation3.step(500)
        gcmc_mover.move(simulation3.context, 200)
        gcmc_mover.report(simulation3)

    # ------------------------------------------------------------------ #
    # Strip ghost waters and write the output PDB                         #
    # ------------------------------------------------------------------ #
    ghost_resids_final = gcmc_mover.getWaterStatusResids(0)
    final_positions = simulation3.context.getState(
        getPositions=True, enforcePeriodicBox=True
    ).getPositions()
    grand.utils.remove_ghosts(
        topology, final_positions,
        ghosts=ghost_resids_final,
        pdb=os.path.join(out_dir, grand_eq_file_name)
    )

    return None
