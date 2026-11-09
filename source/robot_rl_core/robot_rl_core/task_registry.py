"""Discover installed task packages without loading robot runtime assets."""
from importlib import import_module
from importlib.util import find_spec


def register_tasks(task=None):
    packages = []
    # Add future robot packages here using the same Gym/config discovery mechanism.
    for prefix, package in (("d2-", "d2_rl_lab"),):
        if find_spec(package) is not None:
            packages.append(import_module(f"{package}.tasks"))
        elif task and task.startswith(prefix):
            raise ModuleNotFoundError(f"Task {task!r} requires: python -m pip install --no-deps -e source/{package}")
    return packages
