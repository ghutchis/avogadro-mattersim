#  This source file is part of the Avogadro project.
#  This source code is released under the 3-Clause BSD License, (see "LICENSE").

"""Shared EnergyServer loop for MatterSim calculators."""

import sys

import numpy as np
from ase import Atoms
from ase.cell import Cell

from avogadro.energy import EnergyServer

# eV -> kJ/mol
_EV_TO_KJ_MOL = 96.4853321


def run_mattersim_server(mol_cjson: dict, calc) -> None:
    """Run the binary-protocol EnergyServer loop for a MatterSim calculator.

    Args:
        mol_cjson: the "cjson" sub-object from the Avogadro bootstrap JSON.
        calc: an ASE calculator (MatterSimCalculator instance).
    """
    atom_numbers = np.array(mol_cjson["atoms"]["elements"]["number"])
    coord_list = mol_cjson["atoms"]["coords"]["3d"]
    coordinates = np.array(coord_list, dtype=float).reshape(-1, 3)
    # num_atoms must match the atom count C++ uses in every binary frame.
    num_atoms = len(atom_numbers)

    # Avogadro may include periodic image atoms whose fractional coordinates
    # are ~1.0 (equivalent to 0.0 under PBC).  Build a mask that keeps only
    # the unique atoms in the primitive cell.
    keep = np.ones(num_atoms, dtype=bool)

    if "unitCell" in mol_cjson:
        frac_list = mol_cjson["atoms"]["coords"].get("3dFractional", [])
        if frac_list:
            frac = np.array(frac_list, dtype=float).reshape(-1, 3)
            keep = ~np.any(np.abs(frac - 1.0) < 1e-4, axis=1)

        lattice = np.array(
            mol_cjson["unitCell"]["cellVectors"], dtype=float
        ).reshape(3, 3)
        atoms = Atoms(
            atom_numbers[keep], coordinates[keep],
            cell=Cell(lattice), pbc=True,
        )
    else:
        atoms = Atoms(atom_numbers, coordinates, pbc=False)

    atoms.calc = calc

    with EnergyServer(sys.stdin.buffer, sys.stdout.buffer, num_atoms) as server:
        for request in server.requests():
            atoms.set_positions(request.coords[keep])

            if request.wants_energy_and_gradient:
                energy = float(atoms.get_total_energy()) * _EV_TO_KJ_MOL
                forces = atoms.get_forces()
                grad_full = np.zeros((num_atoms, 3), dtype=np.float64)
                grad_full[keep] = forces * -_EV_TO_KJ_MOL
                request.send_energy_and_gradient(energy, grad_full)
            elif request.wants_gradient:
                forces = atoms.get_forces()  # shape (n_unique, 3), eV/A
                grad_full = np.zeros((num_atoms, 3), dtype=np.float64)
                grad_full[keep] = forces * -_EV_TO_KJ_MOL
                request.send_gradient(grad_full)
            else:
                energy = float(atoms.get_total_energy()) * _EV_TO_KJ_MOL
                request.send_energy(energy)
