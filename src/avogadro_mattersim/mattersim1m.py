#  This source file is part of the Avogadro project.
#  This source code is released under the 3-Clause BSD License, (see "LICENSE").

# The MatterSim-1M model is from Microsoft
# https://github.com/microsoft/mattersim
# https://microsoft.github.io/mattersim/

"""MatterSim-1M energy and gradient calculator using the binary protocol."""

import json
import sys
import warnings

import torch
from mattersim.forcefield import MatterSimCalculator

from ._logging import quiet_logging
from ._mattersim_server import run_mattersim_server

# don't use MPS on Apple yet
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run(protocol_out):
    bootstrap = json.loads(sys.stdin.buffer.readline())
    mol_cjson = bootstrap["cjson"]

    quiet_logging()
    warnings.filterwarnings("ignore", category=UserWarning)
    calc = MatterSimCalculator(device=_DEVICE)

    run_mattersim_server(mol_cjson, calc, protocol_out)
