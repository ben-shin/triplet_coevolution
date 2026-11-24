"""Inspect the sdist/wheel, prepare public binary dependencies, then test offline installs."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import venv
import zipfile
from pathlib import Path, PurePosixPath


def members(artifact):
    if artifact.suffix == ".whl":
        with zipfile.ZipFile(artifact) as z:
            paths = z.namelist()
    else:
        with tarfile.open(artifact, "r:gz") as t:
            content = t.getmembers()
            if any(not (m.isfile() or m.isdir()) for m in content):
                raise ValueError("source archive contains an unexpected link or special file")
            paths = [m.name for m in content if m.isfile()]
    for name in paths:
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts or "\\" in name:
            raise ValueError(f"nonportable archive path {name!r}")
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate archive members")
    return paths


def inspect_archives(root, wheel, sdist):
    settings = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf8"))
    examples = {
        name
        for location, items in settings["tool"]["setuptools"]["data-files"].items()
        if "/examples/" in location
        for name in items
    }
    required = {
        p.relative_to(root / "src").as_posix()
        for p in (root / "src/triplet_coevolution").rglob("*.py")
    }
    required |= {
        "triplet_coevolution/reporting/explorer.js",
        "triplet_coevolution/reporting/explorer.css",
    }
    for artifact in (wheel, sdist):
        entries = members(artifact)
        demo = {
            row[row.index("examples/") :]
            for row in entries
            if "/examples/" in row and not row.endswith("/")
        }
        if demo != examples:
            raise ValueError(
                f"{artifact.name} packaged unexpected or missing example data: {sorted(demo ^ examples)}"
            )
        for name in required:
            if not any(row == name or row.endswith("/src/" + name) for row in entries):
                raise ValueError(f"{artifact.name} lacks {name}")
        for row in entries:
            if any(
                part in {"results", ".git", ".venv", "__pycache__"}
                for part in PurePosixPath(row).parts
            ):
                raise ValueError(f"unexpected package content {row!r}")
            if "/input/" in row and not row.endswith("/input/README.md"):
                raise ValueError(f"unexpected raw input in release: {row!r}")


def run(dist, wheelhouse, offline):
    root = Path(__file__).resolve().parents[2]
    wheels = list(dist.glob("*.whl"))
    sources = list(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("supply a directory with one project wheel and one source archive")
    wheel, sdist = wheels[0].resolve(), sources[0].resolve()
    inspect_archives(root, wheel, sdist)
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf8"))
    tools = (root / "requirements/release.txt").read_text().splitlines()
    builds = [line for line in tools if line.startswith(("setuptools==", "wheel=="))]
    wheelhouse = wheelhouse.resolve()
    wheelhouse.mkdir(parents=True, exist_ok=True)
    if not offline:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--only-binary=:all:",
                "--dest",
                str(wheelhouse),
                *project["project"]["dependencies"],
                *builds,
            ],
            check=True,
        )
    print(
        json.dumps(
            {
                "wheelhouse_sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(wheelhouse.glob("*.whl"))
                }
            }
        ),
        flush=True,
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"PYTHONPATH", "PYTHONHOME", "COVERAGE_PROCESS_START"}
    }
    env.update(
        PIP_NO_INDEX="1",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        PIP_CONFIG_FILE=os.devnull,
        PYTHONNOUSERSITE="1",
        PYTHONUTF8="1",
        BLAST_USAGE_REPORT="false",
    )
    with tempfile.TemporaryDirectory(prefix="triplet-offline-install-") as temp:
        for label, archive in (("wheel", wheel), ("sdist", sdist)):
            rootenv = Path(temp) / label
            venv.EnvBuilder(with_pip=True).create(rootenv)
            bindir = rootenv / ("Scripts" if os.name == "nt" else "bin")
            py = bindir / ("python.exe" if os.name == "nt" else "python")
            env["PATH"] = os.pathsep.join((str(bindir), os.environ["PATH"]))
            cwd = Path(temp) / ("run-" + label)
            cwd.mkdir()
            common = [
                str(py),
                "-I",
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-cache-dir",
                "--find-links",
                str(wheelhouse),
            ]
            if label == "sdist":
                subprocess.run([*common, *builds], cwd=cwd, env=env, check=True)
            subprocess.run(
                [*common, "--no-build-isolation", str(archive)], cwd=cwd, env=env, check=True
            )
            subprocess.run([str(py), "-I", "-m", "pip", "check"], cwd=cwd, env=env, check=True)
            subprocess.run(
                [str(py), "-I", str(root / "tools/release/check_installed.py")],
                cwd=cwd,
                env=env,
                check=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", help="Use only the existing wheelhouse")
    args = parser.parse_args()
    run(args.dist, args.wheelhouse, args.offline)
