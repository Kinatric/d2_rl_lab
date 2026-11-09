"""Promote only after all physical checks pass: python promote.py."""
import json
from pathlib import Path
from d2_rl_lab.assets.d2_contract import RUNTIME, asset_manifest, runtime_identity, file_hash

out = Path(__file__).resolve().parents[2] / 'tests/upgrade_isaaclab_3_ea/d2_asset_validation'
manifest = asset_manifest(False)
identity = runtime_identity(manifest)
reports = {}
for backend in ("mujoco", "newton_mjwarp", "isaacsim_physx"):
    report = json.loads((out/(backend+"_standing.json")).read_text())
    assert report.get("status") == "passed" and report["duration_s"] >= 30, backend
    assert report["runtime_identity"] == identity, "Stale standing report"
    reports[backend] = report
receipt = RUNTIME/manifest["asset_identity_sha256"][:12]/"validation/ready_receipt.json"
receipt.parent.mkdir(parents=True, exist_ok=True)
receipt.write_text(json.dumps({"status":"passed", "runtime_identity":identity, "backends":reports},indent=2))
manifest["validation"] = {"status":"ready", "receipt":{"path":receipt.relative_to(RUNTIME).as_posix(),"sha256":file_hash(receipt)}}
manifest["calibration"] = {"status":"ready", "method":"30s_pd_rl_pose_three_backends",
    "standing_base_height_m":reports["mujoco"]["median_height_m"],
    "standing_height_by_backend_m":{key:value["median_height_m"] for key,value in reports.items()}}
(RUNTIME/"asset_manifest.json").write_text(json.dumps(manifest,indent=2))
asset_manifest()
print("D2_ASSETS_READY", identity)
