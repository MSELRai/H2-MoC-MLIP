import numpy as np
from ase.io import read
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    Stationary,
    ZeroRotation,
    MaxwellBoltzmannDistribution,
)
from ase.constraints import FixAtoms
from mace.calculators import MACECalculator


def main():

    #### Inputs ###########
    # init_config must also contain lattice vectors, PBC, etc...
    init_config = "init_conf.xyz" 
    seed = 42  
    temperature = 900  
    nsteps = 1000000  
    dt = 0.5  
    restart = False  

    using_ghost_grid = True  
    ghost_element = "Fr"  
    gres = 5.0  

    freq = 20  
    logfile = "langevin.log"  
    trajfile = "langevin.traj"  

    # Define the potential energy surface
    calc = MACECalculator(
        model_paths=["model_swa.model"],
        device="cuda",
        default_dtype="float32",
    )

    ########### Code starts here ##############
    np.random.seed(seed)
    atoms = read(init_config)
    atoms.calc = calc

    if using_ghost_grid:
        apply_ghost_grid(atoms, ghost_element, gres)

    run_MD(atoms, dt, temperature, trajfile, logfile, nsteps, freq, restart)


def apply_ghost_grid(atoms, ghost_element="Fr", rcut=5.0):
    cell = atoms.get_cell()
    hmat = cell.T
    grid_resolution = np.zeros(3, dtype=int)
    for idx, latvec in enumerate(cell):
        length = np.linalg.norm(latvec)
        ngrid_points = np.ceil(length / rcut)
        grid_resolution[idx] = ngrid_points

    sx = np.linspace(0, 1, grid_resolution[0], endpoint=False)
    sy = np.linspace(0, 1, grid_resolution[1], endpoint=False)
    sz = np.linspace(0, 1, grid_resolution[2], endpoint=False)

    scaled_positions = np.vstack(np.meshgrid(sx, sy, sz)).reshape(3, -1).T
    positions = (hmat @ scaled_positions.T).T

    grid_atoms = Atoms(ghost_element * len(positions), positions)
    ngrid_atoms = len(grid_atoms)

    atoms += grid_atoms

    c = FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == ghost_element])
    atoms.set_constraint(c)


def run_MD(atoms, dt, temperature, trajfile, logfile, nsteps, freq, restart=False):
    if restart:
        dyn = Langevin(
            atoms,
            dt * units.fs,
            temperature_K=temperature,
            friction=0.01 / units.fs,
            logfile=logfile,
            trajectory=trajfile,
            loginterval=freq,
            append_trajectory=True,
        )
        traj = Trajectory(trajfile)
        last_config = traj[-1]
        atoms.set_positions(last_config.get_positions())
        atoms.set_momenta(last_config.get_momenta())
        dyn.nsteps = (len(traj) - 1) * freq

    else:
        dyn = Langevin(
            atoms,
            dt * units.fs,
            temperature_K=temperature,
            friction=0.01 / units.fs,
            logfile=logfile,
            trajectory=trajfile,
            loginterval=freq,
        )
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
        Stationary(atoms)
        ZeroRotation(atoms)

    dyn.run(nsteps)


if __name__ == "__main__":
    main()
