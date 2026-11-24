"""Portable identities for score datasets and their generating software."""

import hashlib
import json
import platform
import re
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np

from triplet_coevolution._revision import ARCHIVE_COMMIT, BUILD_DIRTY, REVISION_SOURCE


def package_code_digest():
    """Hash the installed Python source independently of the current working directory."""
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode() + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def software_identity() -> dict:
    """Identify the installed source and key numerical libraries."""
    package = Path(__file__).resolve().parent
    git = {"commit": None, "dirty": None, "source": "unavailable"}
    root = package.parent.parent
    if package.parent.name == "src" and (root / ".git").exists():
        try:
            cmd = ["git", "-C", str(root)]
            commit = subprocess.check_output(
                [*cmd, "rev-parse", "HEAD"], text=True, timeout=3
            ).strip()
            changed = subprocess.check_output([*cmd, "status", "--porcelain"], timeout=3)
            git = {"commit": commit, "dirty": bool(changed), "source": "source_checkout"}
        except (OSError, subprocess.SubprocessError):
            pass
    elif re.fullmatch(r"[0-9a-f]{40,64}", ARCHIVE_COMMIT):
        git = {"commit": ARCHIVE_COMMIT, "dirty": BUILD_DIRTY, "source": REVISION_SOURCE}
    else:
        direct = metadata.distribution("triplet-coevolution").read_text("direct_url.json")
        commit = json.loads(direct or "{}").get("vcs_info", {}).get("commit_id")
        if commit and re.fullmatch(r"[0-9a-f]{40,64}", commit):
            git = {"commit": commit, "dirty": None, "source": "installed_vcs_metadata"}
    conf = getattr(np.__config__, "CONFIG", {})
    blas = conf.get("Build Dependencies", {}).get("blas", {})
    return {
        "git": git,
        "code_sha256": package_code_digest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "dependencies": {name: metadata.version(name) for name in ("numpy", "biopython", "typer")},
        "blas": {
            name: blas[name]
            for name in ("name", "version", "openblas configuration")
            if name in blas
        },
    }


def file_hashes(directory: Path, *, exclude: tuple[str, ...] = ()) -> dict[str, str]:
    """Hash all published files except named manifests that will be written last."""
    hashes = {}
    for path in sorted(directory.rglob("*")):
        name = path.relative_to(directory).as_posix()
        if path.is_file() and name not in exclude:
            with path.open("rb") as stream:
                hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes
