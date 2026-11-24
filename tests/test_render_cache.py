"""Changing one installed renderer asset must invalidate both public workflow caches."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import triplet_coevolution

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("mode", ["pipeline", "folder"])
def test_installed_renderer_cache(tmp_path, mode):
    # Never modify the developer's or another test's active installation.
    target = tmp_path / "lib" / "triplet_coevolution"
    wheel = os.environ.get("TRIPLET_RENDER_TEST_WHEEL")
    if wheel:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--target",
                str(target.parent),
                wheel,
            ],
            check=True,
        )
        examples = target.parent / "share/triplet-coevolution/examples"
    else:
        shutil.copytree(
            Path(triplet_coevolution.__file__).parent,
            target,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        examples = EXAMPLES
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(target.parent), env.get("PYTHONPATH"))))
    work = tmp_path / "w"
    if mode == "pipeline":
        args = ["pipeline", "--config", examples / "workflow/prepare.json"]
        report_stage = "report"
        upstream = {"prepare", "plan"}
    else:
        args = ["run", "--offline", "--input", examples / "families"]
        report_stage = "family_score"
        upstream = {"family_prepare"}

    def invoke():
        before = set(work.glob("runs/*/pipeline.json"))
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from triplet_coevolution.cli import app; app()",
                *map(str, args),
                "--work-dir",
                str(work),
            ],
            check=True,
            env=env,
            cwd=tmp_path,
            stdout=subprocess.PIPE,
        )
        (manifest,) = set(work.glob("runs/*/pipeline.json")) - before
        stages = {s["stage"]: s for s in json.loads(manifest.read_text(encoding="utf-8"))["stages"]}
        root = Path(stages[report_stage]["output"])
        page = root / ("analysis/explore.html" if mode == "folder" else "explore.html")
        return {k: s["action"] for k, s in stages.items()}, page

    initial_actions, original = invoke()
    assert set(initial_actions.values()) == {"executed"}
    actions, reused = invoke()
    assert set(actions.values()) == {"reused"} and reused == original
    for extension in ("js", "css"):
        asset = target / "reporting" / f"explorer.{extension}"
        pristine = asset.read_bytes()
        marker = f"/* isolated {extension} cache regression */".encode()
        asset.write_bytes(pristine + b"\n" + marker + b"\n")
        actions, regenerated = invoke()
        assert all(actions[name] == "reused" for name in upstream)
        assert actions[report_stage] == "executed" and regenerated != original
        assert marker in regenerated.read_bytes() and marker not in original.read_bytes()
        asset.write_bytes(pristine)
