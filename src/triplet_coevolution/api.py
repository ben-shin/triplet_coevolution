"""Public Python entry points for mixed-alphabet coevolution workflows."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from triplet_coevolution.config import _keys, _typed
from triplet_coevolution.config_layers import load_layers
from triplet_coevolution.operations import file_id, read_json
from triplet_coevolution.parameters import analysis_types

Molecule = Literal["protein", "dna", "rna"]


@dataclass(frozen=True)
class ComponentInput:
    alignment: Path
    molecule: Molecule
    reference_id: str
    reference_start: int = 1
    reference_strand: Literal["+", "-"] = "+"
    reference_row_id: str | None = None
    format: Literal["fasta", "stockholm", "a3m"] | None = None

    def __post_init__(self):
        if not isinstance(self.alignment, (Path, str)) or not str(self.alignment).strip():
            raise ValueError("alignment requires a local file path")
        object.__setattr__(self, "alignment", Path(self.alignment).resolve())
        if self.molecule not in ("protein", "dna", "rna"):
            raise ValueError("molecule must be protein, dna or rna")
        if not isinstance(self.reference_id, str) or not self.reference_id.strip():
            raise ValueError("reference_id requires a nonempty biological identifier")
        if self.reference_row_id is not None and (
            not isinstance(self.reference_row_id, str) or not self.reference_row_id.strip()
        ):
            raise ValueError("reference_row_id requires a nonempty alignment identifier")
        if type(self.reference_start) is not int or self.reference_start < 1:
            raise ValueError("reference_start must be a positive integer")
        if self.reference_strand not in ("+", "-") or (
            self.reference_strand == "-" and self.molecule == "protein"
        ):
            raise ValueError("reference_strand requires + or a nucleotide - strand")
        if self.format not in (None, "fasta", "stockholm", "a3m"):
            raise ValueError("format must be fasta, stockholm or a3m")


def _options(values):
    _keys(values, (), analysis_types(), "analysis")
    result = {}
    for k, v in values.items():
        path = f"analysis.{k}"
        kind = analysis_types()[k]
        if isinstance(v, Path) and kind is Path:
            v = str(v)
        if isinstance(v, tuple) and kind is list:
            v = list(v)
        v = _typed(v, kind, path, Path.cwd())
        if (
            k
            in (
                "min_coverage",
                "max_gap_fraction",
                "max_missing_fraction",
                "identity_threshold",
                "min_overlap",
            )
            and not 0 <= v <= 1
        ):
            raise ValueError(f"{path} must be in [0,1]")
        if (
            k
            in (
                "min_valid",
                "max_pairs",
                "max_triplets",
                "memory_mib",
                "ridge",
                "global_l2",
                "pseudocount",
            )
            and v <= 0
        ):
            raise ValueError(f"{path} must be positive")
        if k in ("min_valid_weight", "benchmark_bootstrap", "permutations", "seed") and v < 0:
            raise ValueError(f"{path} must be nonnegative")
        if isinstance(v, list):
            if not v or any(type(i) is not int or i < 1 for i in v):
                raise ValueError(f"{path} must contain positive integers")
            v = tuple(v)
        result[k] = v
    return MappingProxyType(result)


@dataclass(frozen=True)
class AnalysisRequest:
    a: ComponentInput
    b: ComponentInput
    c: ComponentInput
    matches: Path
    analysis: Mapping[str, object] = field(default_factory=dict)
    config_files: tuple[tuple[Path, str], ...] = field(default=(), repr=False)

    def __post_init__(self):
        if not all(isinstance(c, ComponentInput) for c in (self.a, self.b, self.c)):
            raise TypeError("a, b and c must be ComponentInput instances")
        object.__setattr__(self, "matches", Path(self.matches).resolve())
        if not isinstance(self.analysis, Mapping):
            raise TypeError("analysis must be a mapping of named settings")
        object.__setattr__(self, "analysis", _options(dict(self.analysis)))

    @classmethod
    def from_config(cls, path: Path):
        cfg, chain = load_layers(path)
        _keys(
            cfg,
            {"schema_version", "kind", "components", "matches"},
            {"schema_version", "kind", "components", "matches", "analysis"},
            "analysis configuration",
        )
        if (
            type(cfg["schema_version"]) is not int
            or cfg["schema_version"] != 1
            or cfg["kind"] != "analysis"
        ):
            raise ValueError("expected kind=analysis and integer schema_version=1")
        roles = {}
        _keys(cfg["components"], "ABC", "ABC", "components")
        for role, c in cfg["components"].items():
            name = "components." + role
            _keys(
                c,
                {"molecule", "alignment", "reference_id"},
                ComponentInput.__dataclass_fields__,
                name,
            )
            try:
                roles[role.lower()] = ComponentInput(**c)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name}.{exc}") from exc
        return cls(
            **roles,
            matches=_typed(cfg["matches"], Path, "matches", Path.cwd()),
            analysis=cfg.get("analysis", {}),
            config_files=chain,
        )

    def options(self):
        options = {"matches": self.matches}
        for role in ("a", "b", "c"):
            comp = getattr(self, role)
            options.update(
                {
                    role: comp.alignment,
                    role + "_type": comp.molecule,
                    role + "_ref": comp.reference_id,
                    role + "_start": comp.reference_start,
                    role + "_strand": comp.reference_strand,
                    role + "_format": comp.format,
                    role + "_ref_row": comp.reference_row_id,
                }
            )
        return {
            **options,
            **{k: list(v) if isinstance(v, tuple) else v for k, v in self.analysis.items()},
        }


@dataclass(frozen=True)
class Result:
    directory: Path
    manifest: Path
    summary: dict

    @property
    def report(self) -> Path | None:
        name = self.summary.get("report") or self.summary.get("outputs", {}).get("report")
        return self.directory / name if name else None


def analyze(request: AnalysisRequest, out: Path, *, dry_run: bool = False) -> Result:
    """Score or preflight native data; invalid inputs raise ordinary Python exceptions."""
    from triplet_coevolution.analysis_engine import execute

    if not isinstance(request, AnalysisRequest):
        raise TypeError("request must be an AnalysisRequest")
    directory = Path(out).resolve()
    summary = execute(
        **request.options(), out=directory, dry_run=dry_run, _config_files=request.config_files
    )
    return Result(directory, directory / ("plan.json" if dry_run else "run_summary.json"), summary)


def run(config: Path, out: Path) -> Result:
    """Execute an aligned preparation or raw search workflow with its configured methods."""
    from triplet_coevolution.diagnostics import config_mode
    from triplet_coevolution.workflow import _workflow

    config = Path(config).resolve()
    directory = Path(out).resolve()
    summary = _workflow(config, directory, config_mode(config))
    return Result(directory, directory / "workflow_summary.json", summary)


def run_folders(
    directory: Path,
    out: Path,
    *,
    online: bool | None = None,
    cache_dir: Path | None = None,
    plots: bool | None = None,
) -> Result:
    """Run a three-folder project; the terminal status and scientific exit_code are in summary."""
    from triplet_coevolution.automatic_run import run_families

    output = Path(out).resolve()
    run_families(Path(directory), output, online, cache_dir, plots, announce=False)
    manifest = output / "workflow_summary.json"
    return Result(output, manifest, read_json(manifest))


def select(
    run: Path,
    out: Path,
    *,
    mode: str | None = None,
    maximum: int = 1000,
    seed: int = 0,
    candidate_file: Path | None = None,
    prior_source: str | None = None,
    statistic: str = "mi_bits",
    screen_budget: int = 100000,
    training_ids: list[str] | None = None,
    file_origin: str | None = None,
) -> Result:
    """Recover a verified fitted candidate list, or compute a declared new screening family."""
    import shutil

    from triplet_coevolution.operations import put_json

    run, out = Path(run).resolve(), Path(out).resolve()
    from triplet_coevolution.inference.candidates import select_candidates, write_candidate_manifest
    from triplet_coevolution.inference.scores import coordinate_axes
    from triplet_coevolution.publication import _stage_run
    from triplet_coevolution.validation.null_archive import RefitData, checked_file

    summary = read_json(run / "run_summary.json")
    hashes = summary["artifact_sha256"]
    existing = "candidate_manifest.json" in hashes
    with _stage_run(out) as tmp:
        if not mode and existing:
            if (
                candidate_file
                or training_ids
                or file_origin
                or prior_source
                or statistic != "mi_bits"
                or screen_budget != 100000
                or maximum != 1000
                or seed != 0
            ):
                raise ValueError("choose --mode to request a different screen")
            manifest = checked_file(run, "candidate_manifest.json", hashes)
            filename = read_json(manifest)["coordinate_table"]
            if filename not in hashes or Path(filename).name != filename:
                raise ValueError("unverified candidate coordinate table")
            table = checked_file(run, filename, hashes)
            if file_id(table)["sha256"] != read_json(manifest)["coordinate_table_sha256"]:
                raise ValueError("candidate table checksum differs from manifest")
            shutil.copyfile(manifest, tmp / manifest.name)
            shutil.copyfile(table, tmp / filename)
            action = "recovered_fitted_selection"
        else:
            from triplet_coevolution.io.score_store import load_score_store

            data = RefitData.load(run) if mode in ("entropy", "hybrid") or training_ids else None
            store = (
                data.store
                if data
                else load_score_store(
                    checked_file(run, summary["outputs"]["score_manifest"], hashes)
                )
            )
            if statistic not in store.scores:
                raise ValueError(f"unavailable stored statistic {statistic}")
            chosen = select_candidates(
                store.maps,
                coordinate_axes(store.maps, store.require_reference),
                mode=mode or "uniform",
                maximum=maximum,
                seed=seed,
                candidate_file=candidate_file,
                prior_source=prior_source,
                screen_budget=screen_budget,
                msa=data.msa if data else None,
                weights=data.weights if data else None,
                training_ids=training_ids,
                file_origin=file_origin,
                pair_scores=store.scores[statistic]
                if mode in ("triangle", "wedge", "hybrid") and not training_ids
                else None,
                pair_statistic=statistic,
            )
            write_candidate_manifest(tmp, chosen)
            action = "new_selection"
        put_json(
            tmp / "selection_receipt.json",
            {
                "schema_version": 1,
                "status": "complete",
                "action": action,
                "score_manifest_sha256": hashes[summary["outputs"]["score_manifest"]],
                "candidate_manifest_sha256": file_id(tmp / "candidate_manifest.json")["sha256"],
            },
        )
    manifest = out / "candidate_manifest.json"
    return Result(out, manifest, read_json(manifest))


def benchmark(
    scores: Path,
    labels: Path,
    out: Path,
    *,
    from_v1: bool = False,
    merged: bool = False,
    bootstrap: int = 0,
    seed: int = 0,
    top_k: tuple[int, ...] = (10, 25, 50, 100),
) -> Result:
    """Benchmark the complete common score universe against supplied coordinate labels."""
    from triplet_coevolution.io.score_store import file_digest
    from triplet_coevolution.publication import _stage_run
    from triplet_coevolution.score_views import read_scores, write_json
    from triplet_coevolution.validation.benchmark import benchmark_scores, write_merged_scores

    out = Path(out).resolve()
    labels = Path(labels)
    with _stage_run(out) as stage:
        store = read_scores(Path(scores), from_v1=from_v1)
        result = benchmark_scores(store, labels, top_k=top_k, bootstrap=bootstrap, seed=seed)
        if merged:
            write_merged_scores(stage / "merged_scores.tsv", store, labels)
            if file_digest(labels) != result["label_sha256"]:
                raise ValueError("label input changed between benchmarking and merged export")
        write_json(stage / "benchmark.json", result)
    return Result(out, out / "benchmark.json", result)


def report(
    out: Path,
    *,
    scores: Path | None = None,
    run: Path | None = None,
    from_v1: bool = False,
    labels: Path | None = None,
    mi_score: str | None = None,
    dca_score: str | None = None,
    plots: bool = False,
    max_pairs: int = 1000,
    palette: str = "accessible",
    figure_formats: tuple[str, ...] = (),
    structure: Path | None = None,
) -> Result:
    """Regenerate offline scientific views from verified archives without fitting a model."""
    from triplet_coevolution.io.score_store import file_digest
    from triplet_coevolution.publication import _stage_run
    from triplet_coevolution.report_artifacts import local_artifact
    from triplet_coevolution.score_views import (
        read_scores,
        regenerate_report,
        render_score_view,
        write_json,
        write_ranked_views,
    )

    if (scores is None) == (run is None):
        raise ValueError("choose one of --run and --scores")
    if palette not in ("accessible", "amber-blue") or not set(figure_formats) <= {
        "svg",
        "pdf",
        "png",
    }:
        raise ValueError(
            "palette must be accessible or amber-blue; figure formats are svg, pdf, png"
        )
    out = Path(out).resolve()
    with _stage_run(out) as stage:
        if run is not None:
            if from_v1 or labels or plots or dca_score or mi_score is not None or max_pairs != 1000:
                raise ValueError("use --scores to change score, plot, label or table settings")
            meta = regenerate_report(
                Path(run),
                stage,
                palette=palette,
                figure_formats=figure_formats,
                structure=structure,
            )
            name = "run_summary.json"
        else:
            source = Path(scores)
            manifest = None
            if not from_v1:
                manifest = source
                if source.is_dir():
                    manifest = local_artifact(
                        source,
                        "scores/score_manifest.json"
                        if (source / "scores/score_manifest.json").is_file()
                        else "score_manifest.json",
                    )
                initial = file_digest(manifest)
            store = read_scores(source, from_v1=from_v1)
            tables = write_ranked_views(
                stage, store, mi_score=mi_score, dca_score=dca_score, max_pairs=max_pairs
            )
            meta = render_score_view(
                stage,
                store,
                tables,
                plots=plots,
                labels=labels,
                palette=palette,
                figure_formats=figure_formats,
                structure=structure,
                source_manifest=manifest,
            )
            if manifest and initial != file_digest(stage / "scores/score_manifest.json"):
                raise ValueError("score source changed while rendering")
            name = "view_summary.json"
            write_json(stage / name, meta)
    return Result(out, out / name, meta)
