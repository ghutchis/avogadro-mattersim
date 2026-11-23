#  This source file is part of the Avogadro project.
#  This source code is released under the 3-Clause BSD License, (see "LICENSE").

"""Entry point for the avogadro-mattersim plugin.

Avogadro calls this as:
    avogadro-mattersim <identifier> [--protocol binary-v1] [--lang <locale>] [--debug]

with the molecule bootstrap JSON on stdin (one compact JSON line).
"""

import argparse

from . import _stdio


def main():
    parser = argparse.ArgumentParser("avogadro-mattersim")
    parser.add_argument("feature")
    parser.add_argument("--protocol", default="binary-v1")
    parser.add_argument("--lang", nargs="?", default="en")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # mattersim logs to stdout while importing and while loading weights,
    # which would corrupt the binary frames. Claim the protocol stream and
    # point stdout at stderr before importing it; this must stay ahead of
    # every import below.
    protocol_out = _stdio.claim_stdout()

    match args.feature:
        case "MatterSim-1M":
            from .mattersim1m import run
            run(protocol_out)
        case "MatterSim-5M":
            from .mattersim5m import run
            run(protocol_out)
