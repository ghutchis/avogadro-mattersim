#  This source file is part of the Avogadro project.
#  This source code is released under the 3-Clause BSD License, (see "LICENSE").

# The MatterSim-5M model is from Microsoft
# https://github.com/microsoft/mattersim
# https://microsoft.github.io/mattersim/

import argparse
import json
import sys
import warnings

# copied from cclib.github.io
# 1 eV = 23.060548867 kcal/mol
eV2kcal = 23.060548867

try:
    # for reading the atoms and coordinates
    import numpy as np
    from ase import Atoms
    from ase.cell import Cell

    import torch
    from mattersim.forcefield import MatterSimCalculator
    # don't use MPS on Apple yet
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from loguru import logger

    imported = True
except ImportError:
    imported = False

def getMetaData():
    # before we return metadata, make sure we can import
    if not imported:
        return {}  # Avogadro will ignore us now

    metaData = {
        "name": "MatterSim-5M",
        "identifier": "MatterSim-5M",
        "description": "Calculate MatterSim energies and gradients with 5M model",
        "inputFormat": "cjson",
        "elements": "1-89",
        "unitCell": True,
        "gradients": True,
        "ion": True,
        "radical": True,
    }
    return metaData


def run(filename):
    # we get the molecule from the supplied filename
    #  in cjson format (it's a temporary file created by Avogadro)
    with open(filename, "r") as f:
        mol_cjson = json.load(f)

    # first setup the calculator
    atoms = np.array(mol_cjson["atoms"]["elements"]["number"])
    coord_list = mol_cjson["atoms"]["coords"]["3d"]
    coordinates = np.array(coord_list, dtype=float).reshape(-1, 3)

    # set up the ASE Atoms object
    atoms = Atoms(atoms, coordinates)
    # presumably we have a unit cell
    if 'unitCell' in mol_cjson:
        cell = mol_cjson['unitCell']
        lattice = np.array(cell['cellVectors']).reshape(3, 3)
        atoms.cell = Cell(lattice)
        atoms.pbc = True
    else:
        atoms.pbc = False

    # disable logger
    logger.remove(0)
    warnings.filterwarnings("ignore", category=UserWarning)
    calc = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=device)
    atoms.calc = calc

    # we loop forever - Avogadro will kill the process when done
    while(True):
        # read new coordinates from stdin
        for i in range(len(atoms)):
            items = input().split()
            if len(items) == 3:
                for j in range(len(items)):
                    coordinates[i][j] = float(items[j])

        # update the calculator and run a new calculation
        atoms.set_positions(coordinates)

        # first print the energy of these coordinates
        # in kcal/mol
        print("AvogadroEnergy:", atoms.get_potential_energy() * eV2kcal)

        # now print the gradient
        # .. we don't want the "[]" in the output
        print("AvogadroGradient:")
        grad = atoms.get_forces() * -eV2kcal  # convert units
        output = np.array2string(grad)
        output = output.replace("[", "").replace("]", "")
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MatterSim 1M calculator")
    parser.add_argument("--display-name", action="store_true")
    parser.add_argument("--metadata", action="store_true")
    parser.add_argument("-f", "--file", nargs=1)
    parser.add_argument("--lang", nargs="?", default="en")
    args = vars(parser.parse_args())

    if args["metadata"]:
        print(json.dumps(getMetaData()))
    elif args["display_name"]:
        name = getMetaData().get("name")
        if name:
            print(name)
        else:
            sys.exit("MatterSim is unavailable")
    elif args["file"]:
        run(args["file"][0])
