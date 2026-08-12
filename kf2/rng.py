"""Deterministic random number generation.

Two properties this module exists to provide:

1. **Named, independent streams per noise source.** Common random numbers
   require that swapping the estimator cannot perturb the measurement sequence.
   Named streams also mean that adding a noise source in a later phase (clutter,
   detection, sensor bias) does not shift the streams that already exist, so
   results recorded today still reproduce afterwards. Never renumber `Stream`.

2. **Order independence.** A stream is a pure function of
   ``(seed, run index, stream id)`` -- there is no global state and no
   sequential consumption across runs, so runs may be executed in any order, in
   parallel, or individually, and still produce identical data.

What is *not* claimed: bit-identical output across numpy versions. NumPy's
policy (NEP 19) allows `Generator`'s stream to change between releases. The
version is pinned in requirements.txt, and that is the honest scope of the
reproducibility claim -- "reproducible from a seed and a scenario file, on a
pinned environment".
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class Stream(IntEnum):
    """Noise sources. Append only -- renumbering breaks recorded seeds."""

    PROCESS = 1
    MEASUREMENT = 2
    INIT = 3
    CLUTTER = 4
    DETECTION = 5
    TARGET_MANOEUVRE = 6


def stream_rng(seed: int, run: int, stream: Stream) -> np.random.Generator:
    """Return the generator for one (seed, run, stream) triple.

    Uses SeedSequence spawn keys rather than hashing the inputs into a single
    integer: spawn keys are designed for exactly this, and give streams with no
    detectable correlation between them.
    """
    if run < 0:
        raise ValueError("run index must be >= 0")
    ss = np.random.SeedSequence(entropy=int(seed), spawn_key=(int(run), int(stream)))
    return np.random.default_rng(ss)
