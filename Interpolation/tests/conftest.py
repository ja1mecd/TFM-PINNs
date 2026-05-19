import sys
from pathlib import Path

# Interpolation scripts import each other as top-level modules
# (e.g. `from pinn_interpolant_l2 import ...`), so the package dir
# must be on sys.path for the test process.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
