"""Protocol tests for avogadro-mattersim.

These run the real entry point in a subprocess and speak the same binary
protocol Avogadro does. The point is to catch anything written to stdout
(mattersim logs through loguru to stdout while it imports and loads weights)
ahead of or between response frames: Avogadro requires every response to start
with the AVB1 magic and does not resynchronize. The first run downloads the
model weights.
"""

import io
import json
import struct
import subprocess
import sys

import numpy as np
import pytest

from avogadro.energy import (
    FLAG_REQUEST_ENERGY_AND_GRADIENT,
    FLAG_REQUEST_GRADIENT,
    read_frame,
    write_frame,
)

# python -c leaves sys.argv[1:] as the arguments after the code.
_LAUNCH = [sys.executable, "-c", "from avogadro_mattersim import main; main()"]

_WATER = np.array(
    [[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]]
)

# Rock salt, primitive fcc cell.
_A = 5.64
_LATTICE = _A / 2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
# Na, Cl, then an Avogadro-style image of Na at fractional 1.0.
_FRAC = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1.0, 0, 0]])
_CRYSTAL = _FRAC @ _LATTICE


def _molecule_bootstrap():
    cjson = {
        "atoms": {
            "elements": {"number": [8, 1, 1]},
            "coords": {"3d": _WATER.reshape(-1).tolist()},
        }
    }
    return {"cjson": cjson, "charge": 0, "spin": 1}


def _crystal_bootstrap():
    cjson = {
        "atoms": {
            "elements": {"number": [11, 17, 11]},
            "coords": {
                "3d": _CRYSTAL.reshape(-1).tolist(),
                "3dFractional": _FRAC.reshape(-1).tolist(),
            },
        },
        "unitCell": {"cellVectors": _LATTICE.reshape(-1).tolist()},
    }
    return {"cjson": cjson, "charge": 0, "spin": 1}


def _request(coords, flags=0):
    buffer = io.BytesIO()
    write_frame(
        buffer, flags, len(coords), np.ascontiguousarray(coords, "<f8").tobytes()
    )
    return buffer.getvalue()


def _run(identifier, bootstrap, base, atom, axis):
    """Run four requests and return (plus, minus, gradient, energy, fused)."""
    step = 0.01
    shifted = []
    for sign in (+1, -1):
        coords = base.copy()
        coords[atom, axis] += sign * step
        shifted.append(coords)

    stdin = json.dumps(bootstrap).encode() + b"\n"
    stdin += _request(shifted[0])
    stdin += _request(shifted[1])
    stdin += _request(base, FLAG_REQUEST_GRADIENT)
    stdin += _request(base, FLAG_REQUEST_ENERGY_AND_GRADIENT)

    process = subprocess.run(
        _LAUNCH + [identifier, "--protocol", "binary-v1"],
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=900,
    )
    assert process.returncode == 0, process.stderr.decode("utf-8", "replace")

    # The very first bytes must be a frame, not a log line from mattersim.
    assert process.stdout[:4] == b"AVB1", process.stdout[:200]

    n = len(base)
    stream = io.BytesIO(process.stdout)
    plus = struct.unpack("<d", read_frame(stream).payload)[0]
    minus = struct.unpack("<d", read_frame(stream).payload)[0]
    gradient = np.frombuffer(read_frame(stream).payload, "<f8").reshape(n, 3)
    fused = read_frame(stream).payload
    energy = struct.unpack("<d", fused[:8])[0]
    fused_gradient = np.frombuffer(fused[8:], "<f8").reshape(n, 3)
    assert not stream.read(), "unexpected bytes after the last frame"

    assert np.isfinite([plus, minus, energy]).all()
    assert np.all(np.isfinite(gradient))
    np.testing.assert_allclose(fused_gradient, gradient, atol=1e-3)
    return plus, minus, gradient, step


@pytest.mark.parametrize("identifier", ["MatterSim-1M", "MatterSim-5M"])
def test_molecule_frames_only_and_gradient_matches_energy(identifier):
    plus, minus, gradient, step = _run(
        identifier, _molecule_bootstrap(), _WATER, 1, 1
    )
    # Avogadro wants dE/dx in kJ/mol/Angstrom; compare with finite differences.
    numeric = (plus - minus) / (2 * step)
    assert gradient[1, 1] == pytest.approx(numeric, rel=0.1, abs=5.0)


@pytest.mark.parametrize("identifier", ["MatterSim-1M", "MatterSim-5M"])
def test_crystal_with_image_atom(identifier):
    # Displace Cl (index 1); the image atom (index 2) is filtered out of the
    # calculation, so its gradient is exactly zero.
    plus, minus, gradient, step = _run(
        identifier, _crystal_bootstrap(), _CRYSTAL, 1, 0
    )
    numeric = (plus - minus) / (2 * step)
    assert gradient[1, 0] == pytest.approx(numeric, rel=0.1, abs=5.0)
    np.testing.assert_array_equal(gradient[2], 0.0)
