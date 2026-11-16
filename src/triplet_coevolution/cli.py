
# run the matched triplet mi pipeline from terminal

from pathlib import Path

import typer

from triplet_coevolution.core.coordinates import build_reference_map, subset_reference_map
from triplet_coevolution.inference.mi import triplet_mi
from triplet_coevolution.inference.weights import sequence_weights
from triplet_coevolution.io.alignments import read_msa
from triplet_coevolution.io.export import write_triplet_mi_tables
from triplet_coevolution.io.metadata import read_triplet_table
from triplet_coevolution.msa.masking import apply_triplet_masks, column_mask
from triplet_coevolution.msa.qc import triplet_qc
from triplet_coevolution.msa.synchronize import synchronize


app = typer.Typer(help="three component coev analysis", no_args_is_help=True)


@app.callback()
def main() -> None:
  """analyze verified matched msas"""


@app.command("analyze")
def analyze(
  a: Path = typer.Option(..., "--a", help="aligned A fasta or stockholm"),
  b: Path = typer.Option(..., "--b", help="aligned B fasta or stockholm"),
  c: Path = typer.Option(..., "--c", help="aligned C fasta or stockholm"),
  matches: Path = typer.Option(..., "--matches", help="curated triplet csv or tsv"),
  a_ref: str = typer.Option(..., "--a-ref", help="A ref seq id"),
  b_ref: str = typer.Option(..., "--b-ref", help="B ref seq id"),
  c_ref: str = typer.Option(..., "--c-ref", help="C ref seq id"),
  out: Path = typer.Option(Path("results"), "--out", help="output dir"),
  a_type: str = typer.Option("protein", "--a-type"),
  b_type: str = typer.Option("protein", "--b-type"),
  c_type: str = typer.Option("protein", "--c-type"),
  a_start: int = typer.Option(1, "--a-start"),
  b_start: int = typer.Option(1, "--b-start"),
  c_start: int = typer.Option(1, "--c-start"),
  min_coverage: float = typer.Option(0.5, "--min-coverage"),
  max_gap_fraction: float = typer.Option(0.5, "--max-gap-fraction"),
  max_missing_fraction: float = typer.Option(0.5, "--max-missing-fraction"),
  identity_threshold: float = typer.Option(0.8, "--identity-threshold"),
  min_overlap: float = typer.Option(0.5, "--min-overlap"),
  min_valid: int = typer.Option(2, "--min-valid"),
  max_pairs: int = typer.Option(1000, "--max-pairs"),
  score: str = typer.Option("mi_apc", "--score"),
  fmt: str = typer.Option("tsv", "--fmt"),
  require_reference: bool = typer.Option(False, "--require-reference"),
  plots: bool = typer.Option(False, "--plots/--no-plots"),
) -> None:
  """analyze three aligned msas using curated homolog matches"""

  try:
    # validate output opts before reading data
    if score not in {"mi_bits", "mi_apc"}:
      raise ValueError("score must be mi_bits or mi_apc")
    if fmt not in {"csv", "tsv"}:
      raise ValueError("fmt must be csv or tsv")
    if min_valid < 1 or max_pairs < 1:
      raise ValueError("min_valid and max_pairs must be positive ints")

    # only need matplotlib when plots are requested
    if plots:
      from triplet_coevolution.reporting.plots import save_mi_plots

    # read alignments and establish ref numbering before masking
    original = (
      read_msa(a, "A", a_type),
      read_msa(b, "B", b_type),
      read_msa(c, "C", c_type),
    )
    maps = tuple(
      build_reference_map(comp, ref_id, start)
      for comp, ref_id, start in zip(
        original, (a_ref, b_ref, c_ref), (a_start, b_start, c_start)
      )
    )

    # only verified triplets become matched obs
    table = read_triplet_table(matches)
    matched = synchronize(*original, table)

    # remove unreliable cols separately for each comp
    qc = triplet_qc(matched)
    keep = tuple(
      column_mask(
        part,
        min_coverage=min_coverage,
        max_gap_fraction=max_gap_fraction,
        max_missing_fraction=max_missing_fraction,
      )
      for part in qc.components
    )
    if any(not mask.any() for mask in keep):
      raise ValueError("qc removed every col in at least one comp")

    filtered = apply_triplet_masks(matched, *keep)
    maps = tuple(
      subset_reference_map(mapping, cols)
      for mapping, cols in zip(
        maps, (filtered.a_columns, filtered.b_columns, filtered.c_columns)
      )
    )

    # weight matched obs and calculate all three mi matrices
    weights = sequence_weights(
      filtered.msa,
      identity_threshold=identity_threshold,
      min_overlap=min_overlap,
    )
    result = triplet_mi(filtered.msa, weights=weights, min_valid=min_valid)

    # export tables w original aln and ref positions
    tables = write_triplet_mi_tables(
      out, result, *maps,
      score=score,
      max_pairs=max_pairs,
      min_valid=min_valid,
      require_reference=require_reference,
      fmt=fmt,
    )

    if plots:
      save_mi_plots(out / "figures", result, *maps, score=score, min_valid=min_valid)

  except (ValueError, TypeError, OSError, ImportError) as exc:
    typer.echo(f"error: {exc}", err=True)
    raise typer.Exit(code=1) from exc

  typer.echo(f"matched obs: {matched.n_observations}")
  typer.echo(f"effective depth: {weights.n_eff:.2f}")
  typer.echo(
    f"kept cols: A={filtered.msa.a.n_sites} "
    f"B={filtered.msa.b.n_sites} C={filtered.msa.c.n_sites}"
  )
  for name, path in tables.items():
    typer.echo(f"{name}: {path}")
  if plots:
    typer.echo(f"plots: {out / 'figures'}")


if __name__ == "__main__":
  app()
