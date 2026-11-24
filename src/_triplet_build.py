"""Carry source-checkout Git identity from sdist into wheels built outside Git."""

import subprocess
from pathlib import Path

from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


def _write_build_revision(destination: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        return  # git archive substitution or the original sdist already records its source
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=5
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], timeout=5)
        )
    except (OSError, subprocess.SubprocessError):
        return
    # setuptools may hard-link the release tree. Replacing the file keeps the
    # tracked checkout intact when stamping the distribution.
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(
        f'ARCHIVE_COMMIT = "{revision}"\nBUILD_DIRTY = {dirty!r}\n'
        'REVISION_SOURCE = "build_checkout"\n',
        encoding="utf-8",
    )
    temporary.replace(destination)


class BuildPy(build_py):
    def run(self) -> None:
        super().run()
        if not self.editable_mode:
            _write_build_revision(Path(self.build_lib) / "triplet_coevolution" / "_revision.py")


class SourceDist(sdist):
    def make_release_tree(self, base_dir, files) -> None:
        super().make_release_tree(base_dir, files)
        _write_build_revision(Path(base_dir) / "src" / "triplet_coevolution" / "_revision.py")
