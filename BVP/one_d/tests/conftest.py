import sys
from pathlib import Path

# BVP one_d scripts import each other as top-level modules
# (e.g. `from activation_stats_bvp import ...`), so the package dir
# must be on sys.path for the test process.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
