"""Atomic artifact publication and native input identity checks."""

import csv
import hashlib
import shutil
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path

from triplet_coevolution.command_progress import event

SUMMARY_NAME = "run_summary.json"


def _check_output(out: Path) -> None:
    if (out.exists() or out.is_symlink()) and not out.is_dir():
        raise ValueError(f"--out must be a new or empty directory: {out}")
    if out.is_dir() and any(out.iterdir()):
        raise ValueError(f"output directory is not empty: {out}. Choose a new --out directory.")


@contextmanager
def _stage_run(out: Path) -> Iterator[Path]:
    """Complete the run in private and publish its manifest last.

    The exclusive staging directory serializes writers targeting the same output.
    Exclusive file creation and rollback also protect pre-existing empty directories.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        out.mkdir()
        created = True
    except FileExistsError:
        _check_output(out)

    stage = out / ".triplet-in-progress"
    try:
        stage.mkdir()
        try:
            if any(path != stage for path in out.iterdir()):
                raise ValueError(
                    f"output directory is not empty: {out}. Choose a new --out directory."
                )
            event("stage_started", output=str(out))
            yield stage
            with ExitStack() as rollback:
                # Directories come before their contents. The manifest marks a completed run.
                marker_names = (
                    "shard_plan.json",
                    "shard_receipt.json",
                    "selection_receipt.json",
                    "validation.json",
                    "doctor.json",
                    "stability_summary.json",
                    "conditional_summary.json",
                    "dca_comparison.json",
                    "split_manifest.json",
                    "calibration.json",
                    "structure_summary.json",
                    "workflow_summary.json",
                    SUMMARY_NAME,
                    "view_summary.json",
                    "benchmark.json",
                    "score_manifest.json",
                    "plan.json",
                )
                final_manifest = next(
                    (stage / name for name in marker_names if (stage / name).exists()), None
                )
                artifacts = sorted(
                    stage.rglob("*"), key=lambda p: (p == final_manifest, p.name in marker_names, p)
                )
                for artifact in artifacts:
                    target = out / artifact.relative_to(stage)
                    if artifact.is_dir():
                        target.mkdir()
                        rollback.callback(target.rmdir)
                    else:
                        with artifact.open("rb") as source, target.open("xb") as dest:
                            rollback.callback(target.unlink)
                            shutil.copyfileobj(source, dest)
                rollback.pop_all()
                event("stage_published", output=str(out), files=sum(p.is_file() for p in artifacts))
        finally:
            shutil.rmtree(stage)
    except BaseException:
        event("stage_failed", output=str(out))
        if created:
            # A concurrent writer may have claimed this destination.
            with suppress(OSError):
                out.rmdir()
        raise


def _file_stamp(path: Path) -> tuple[int, int, int, int]:
    if not path.is_file():
        raise ValueError(f"input file not found: {path}")
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _input_identity(path: Path) -> dict[str, str]:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path.absolute()), "sha256": digest}


def _row_count(path: Path, fmt: str) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.reader(handle, delimiter="\t" if fmt == "tsv" else ",")
        next(rows)  # Header is present even when there are no qualifying pairs.
        return sum(1 for _ in rows)
