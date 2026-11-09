"""Resolve the repository without depending on a server path or output depth."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in HERE.parents if (p / 'd2_rl_lab.sh').is_file()), None)
if ROOT is None:
    raise RuntimeError('Keep the experiment directory inside the D2_rl_lab checkout.')
