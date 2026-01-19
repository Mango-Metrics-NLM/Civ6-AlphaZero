import numpy as np
import sys

print(f"NumPy version: {np.__version__}")

try:
    import numpy.core.multiarray as ma1
    print(f"numpy.core.multiarray.scalar: {ma1.scalar}")
except Exception as e:
    print(f"numpy.core.multiarray failed: {e}")

try:
    import numpy._core.multiarray as ma2
    print(f"numpy._core.multiarray.scalar: {ma2.scalar}")
except Exception as e:
    print(f"numpy._core.multiarray failed: {e}")
