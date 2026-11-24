"""Run in an isolated environment outside the checkout using its installed distribution."""

import hashlib
import json
import subprocess
import sys
import sysconfig
from html.parser import HTMLParser
from importlib import import_module, metadata, resources
from pathlib import Path


class OfflineAssets(HTMLParser):
    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        if (
            tag in {"script", "img", "iframe", "source"}
            or tag == "link"
            and attr.get("rel") == "stylesheet"
        ):
            url = attr.get("src") or attr.get("href") or ""
            assert not url.startswith(("https:", "http:", "//")), f"remote report asset: {url}"


def main():
    package = import_module("triplet_coevolution")
    assert Path(package.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    version = metadata.version("triplet-coevolution")
    assert package.__version__ == version
    for name in (
        "core.coordinates",
        "inference.mi",
        "inference.potts",
        "inference.threebody",
        "validation.null_runner",
        "io.score_store",
        "search.discovery",
        "structure.mapping",
        "validation.shards",
        "reporting",
    ):
        import_module(f"triplet_coevolution.{name}")
    assets = resources.files("triplet_coevolution.reporting")
    for asset in ("explorer.js", "explorer.css"):
        assert assets.joinpath(asset).read_bytes(), f"missing report asset {asset}"
    data = Path(sysconfig.get_path("data")) / "share/triplet-coevolution"
    assert (data / "input/README.md").is_file()
    assert (data / "examples/hpc/null-array.sh").is_file()
    for name in (
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "docs/release-engineering.md",
        "benchmarks/gatabc_manifest.json",
        "benchmarks/independent_evaluation_policy.json",
        "benchmarks/independent_contact_metrics.json.gz",
        "benchmarks/independent_contact_metrics.tsv",
        "benchmarks/independent_nested_metrics.json",
        "benchmarks/experimental_fluorescence_results.json",
        "benchmarks/verify_independent_contacts.py",
        "benchmarks/verify_independent_nested.py",
        "docs/implementation-report.md",
    ):
        assert (data / name).is_file(), name
    exe = Path(sysconfig.get_path("scripts")) / (
        "triplet.exe" if sys.platform == "win32" else "triplet"
    )

    def run(*args):
        return subprocess.check_output([str(exe), *map(str, args)], text=True, encoding="utf8")

    assert version in run("--version")
    assert "analyze" in run("--help")
    mixed = data / "examples/mixed"
    run(
        "analyze",
        "--a",
        mixed / "protein.fasta",
        "--b",
        mixed / "dna.fasta",
        "--c",
        mixed / "rna.fasta",
        "--b-type",
        "dna",
        "--c-type",
        "rna",
        "--matches",
        mixed / "matches.tsv",
        "--a-ref",
        "prot_ref",
        "--b-ref",
        "dna_ref",
        "--c-ref",
        "rna_ref",
        "--out",
        "mixed",
    )
    src = Path("mixed")
    info = json.loads((src / "run_summary.json").read_text(encoding="utf8"))
    assert info["matched_observations"] == 8
    payload = src / info["outputs"]["score_payload"]
    before = hashlib.sha256(payload.read_bytes()).hexdigest()
    run("report", "--run", src, "--out", "regenerated")
    assert (Path("regenerated") / "report.html").is_file()
    explorer = (Path("regenerated") / "explore.html").read_text(encoding="utf8")
    OfflineAssets().feed(explorer)
    assert hashlib.sha256(payload.read_bytes()).hexdigest() == before
    run("run", "--config", data / "examples/workflow/run.json", "--out", "workflow", "--no-plots")
    workflow = Path("workflow")
    info = json.loads((workflow / "workflow_summary.json").read_text(encoding="utf8"))
    assert info["matched_groups"] == 8 and (workflow / info["report"]).is_file()
    print(
        json.dumps(
            {
                "version": version,
                "mixed_observations": 8,
                "workflow_observations": 8,
                "score_payload_sha256": before,
                "offline_report": info["report"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
