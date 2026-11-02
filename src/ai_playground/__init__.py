"""AI Playground."""

import os

# NumPy's OpenBLAS starts one thread per logical core. CharNet's matrices are tiny, so on a
# 16-thread machine with any background load a read slowed from ~2 ms to ~35-50 ms; one thread
# does it in ~2 ms and leaves the cores to MediaPipe, the camera, and the render loop.
# OpenBLAS reads this once when NumPy loads, so it has to be set before the first numpy import.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
