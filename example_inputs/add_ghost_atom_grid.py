import os
import numpy as np
from ase import Atoms
from ase.io import iread, write
from ase.constraints import FixAtoms

########## Inputs ########

traj_file = "db.xyz"
outfile = "db_grid.xyz"
ghost_element = "Fr"
rcut = 5.0

########### CODE STARTS HERE ########3
try:
    os.remove(outfile)
except OSError:
    pass

traj = iread(traj_file)

for atoms in traj:
    cell = atoms.get_cell()
    hmat = cell.T
    grid_resolution = np.zeros(3, dtype = int)
    for idx, latvec in enumerate(cell):
        length = np.linalg.norm(latvec)
        ngrid_points = np.ceil(length / rcut)
        grid_resolution[idx] = ngrid_points

    sx = np.linspace(0, 1, grid_resolution[0], endpoint = False)
    sy = np.linspace(0, 1, grid_resolution[1], endpoint = False)
    sz = np.linspace(0, 1, grid_resolution[2], endpoint = False)

    scaled_positions = np.vstack(np.meshgrid(sx,sy,sz)).reshape(3,-1).T
    positions = (hmat @ scaled_positions.T).T

    grid_atoms = Atoms(ghost_element * len(positions), positions)
    ngrid_atoms = len(grid_atoms)

    atoms += grid_atoms

    c = FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == ghost_element])
    atoms.set_constraint(c)

    write(outfile, atoms, append=True)
