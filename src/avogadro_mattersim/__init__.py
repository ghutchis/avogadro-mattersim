#  This source file is part of the Avogadro project.
#  This source code is released under the 3-Clause BSD License, (see "LICENSE").

"""Entry point for the avogadro-mattersim plugin.

Avogadro calls this as:
    avogadro-mattersim <identifier> [--protocol binary-v1] [--lang <locale>] [--debug]

with the molecule bootstrap JSON on stdin (one compact JSON line).
"""

import argparse


def main():
    parser = argparse.ArgumentParser("avogadro-mattersim")
    parser.add_argument("feature")
    parser.add_argument("--protocol", default="binary-v1")
    parser.add_argument("--lang", nargs="?", default="en")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    match args.feature:
        case "MatterSim-1M":
            from .mattersim1m import run
            run()
        case "MatterSim-5M":
            from .mattersim5m import run
            run()
