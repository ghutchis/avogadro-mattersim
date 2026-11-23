#  This source file is part of the Avogadro project.
#  This source code is released under the 3-Clause BSD License, (see "LICENSE").

"""Keep the binary protocol stream clean.

Avogadro requires every response to begin with the AVB1 magic and does not
resynchronize, so a single stray byte on stdout -- a library announcing itself
on import, a download notice, a C extension writing straight to file
descriptor 1 -- breaks the session. Redirecting sys.stdout only catches Python
code, so this works at the file-descriptor level instead.
"""

import os
import sys


def claim_stdout():
    """Return a binary stream for protocol frames, and divert everything else.

    Call this once, before importing anything heavy. The original stdout is
    duplicated to a private descriptor that only the returned stream uses, then
    descriptor 1 is pointed at stderr, so anything else written to "stdout" --
    print(), C or C++ output, subprocesses -- lands on stderr instead.

    The stream is buffered; EnergyServer flushes after every frame.

    os.dup and os.dup2 behave the same on Windows. The one difference is that a
    process started without a console (for instance by pythonw) may have no
    valid descriptor 1 or 2; there is nothing to protect then, so this falls
    back to sys.stdout.buffer.
    """
    try:
        sys.stdout.flush()
        protocol = os.fdopen(os.dup(1), "wb")
        os.dup2(2, 1)
    except (OSError, ValueError, AttributeError):
        return sys.stdout.buffer
    return protocol
