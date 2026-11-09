#!/usr/bin/env python3
"""Relocate manifest input paths without changing robot/policy identity.

Run in an isolated checkout: python scripts/tools/migrate_d2_asset_paths.py
The original manifest is retained beside the migrated one. No model is rewritten.
"""
import copy
import json
import os
from pathlib import Path

from d2_rl_lab.assets.d2_contract import (
    DESCRIPTION, RUNTIME, file_hash, policy_identity, runtime_identity, validate_manifest,
)


def main():
    path = RUNTIME / "asset_manifest.json"
    raw = path.read_bytes()
    original = json.loads(raw)
    candidate = copy.deepcopy(original)
    relative = Path(os.path.relpath(DESCRIPTION, RUNTIME)).as_posix()
    candidate["source"]["root"] = relative
    members = {"urdf": "urdf/d2_description.urdf", "motor_yaml": "config/robstride06_motor.yaml",
               "joint_adapter": "config/joint_adapter.yaml"}
    for key, member in members.items():
        if file_hash(DESCRIPTION / member) != original["inputs"][key]["sha256"]:
            raise ValueError(f"Source content differs: {member}")
        candidate["inputs"][key]["path"] = relative + "/" + member
    # Validate all source/output dependencies and the standing receipt before writing.
    validate_manifest(candidate)
    if runtime_identity(original) != runtime_identity(candidate):
        raise ValueError("Runtime identity changed")
    if policy_identity(manifest=original) != policy_identity(manifest=candidate):
        raise ValueError("Policy identity changed")
    if candidate == original:
        print("Already migrated; identity and all dependencies verified.")
        return
    backup = path.with_suffix(".json.before_dual_package")
    if backup.exists() and backup.read_bytes() != raw:
        raise FileExistsError(f"Refusing to overwrite a different backup: {backup}")
    if not backup.exists():
        backup.write_bytes(raw)
    temporary = path.with_suffix(".json.migrating")
    temporary.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print("Migrated source paths; source, runtime and policy identities unchanged.")


if __name__ == "__main__":
    main()
