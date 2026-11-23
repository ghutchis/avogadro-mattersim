#  This source file is part of the Avogadro project.
#  This source code is released under the 3-Clause BSD License, (see "LICENSE").

"""Quiet mattersim's loguru logging.

mattersim installs a loguru handler on sys.stdout when it is imported
(forcefield/potential.py calls get_logger()), and logs INFO lines to it while
loading a model. Handler ids are not stable, so remove everything and log
warnings to stderr instead.
"""

import sys

from loguru import logger


def quiet_logging():
    """Replace every loguru handler with one that writes warnings to stderr.

    Call after importing mattersim, since the import adds its own handler.
    """
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
