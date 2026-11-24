"""Ranked views of complete MI and DCA scores with mapped coordinates."""

import csv
from collections.abc import Iterator
from numbers import Integral
from pathlib import Path

import numpy as np

from triplet_coevolution.core.coordinates import ReferenceCoordinates
from triplet_coevolution.inference.mi import MIBlock, TripletMI
from triplet_coevolution.inference.potts import DCABlock
from triplet_coevolution.inference.scores import Method, coordinate_axes, get_score, ranked_indices

FIELDS = (
  "rank", "competition_rank", "statistic", "direction", "score",
  "pair",
  "x_component",
  "x_reference_id",
  "x_col",
  "x_original_col",
  "x_reference_pos",
  "x_symbol",
  "y_component",
  "y_reference_id",
  "y_col",
  "y_original_col",
  "y_reference_pos",
  "y_symbol",
  "mi_bits",
  "mi_apc",
  "n_valid",
  "valid_weight",
)


DCA_FIELDS = tuple(
  {"mi_bits": "dca_frobenius", "mi_apc": "dca_apc"}.get(key, key) for key in FIELDS
)


def _check_options(
  score: str,
  max_pairs: int | None,
  min_valid: int,
  require_reference: bool,
  method: Method = "mi",
) -> None:
  """check ranking and export options"""

  get_score(score, method)

  if max_pairs is not None and (
    isinstance(max_pairs, bool)
    or not isinstance(max_pairs, Integral)
    or max_pairs < 1
  ):
    raise ValueError("max_pairs must be a positive int or None")

  if (
    isinstance(min_valid, bool)
    or not isinstance(min_valid, Integral)
    or min_valid < 1
  ):
    raise ValueError("min_valid must be a positive int")

  if not isinstance(require_reference, bool):
    raise TypeError("require_reference must be boolean")


def _check_block(
  block: MIBlock,
  x_map: ReferenceCoordinates,
  y_map: ReferenceCoordinates,
) -> None:
  """check mi matrix and ref maps refer to matching comps and cols"""

  if not isinstance(block, MIBlock):
    raise TypeError("block must be an MIBlock")

  if not isinstance(x_map, ReferenceCoordinates) or not isinstance(y_map, ReferenceCoordinates):
    raise TypeError("x_map and y_map must be ReferenceCoordinates")

  if (block.component_x, block.component_y) != (x_map.component, y_map.component):
    raise ValueError("mi components and reference maps dont match")

  expected = (x_map.n_sites, y_map.n_sites)

  for name in ("mi_bits", "mi_apc", "n_valid", "valid_weight"):
    if np.asarray(getattr(block, name)).shape != expected:
      raise ValueError(f"{name} shape doesnt match reference maps")

  if (
    np.isinf(block.mi_bits).any()
    or np.isinf(block.mi_apc).any()
    or np.any(np.isfinite(block.mi_bits) & (block.mi_bits < 0))
  ):
    raise ValueError("mi scores must be finite or nan; raw mi cant be negative")

  missing_support = block.n_valid == -1
  if (block.n_valid.dtype.kind not in "iu" or np.any(block.n_valid < -1)
      or np.any(missing_support & (np.isfinite(block.mi_bits) | np.isfinite(block.mi_apc)))):
    raise ValueError("n_valid requires nonnegative counts or unknown support on omitted legacy cells")

  if (not np.array_equal(np.isnan(block.valid_weight), missing_support)
      or np.isinf(block.valid_weight).any() or np.any(block.valid_weight < 0)):
    raise ValueError("valid_weight must be finite and nonnegative")


def ranked_mi_rows(
  block: MIBlock,
  x_map: ReferenceCoordinates,
  y_map: ReferenceCoordinates,
  *,
  score: str = "mi_apc",
  max_pairs: int | None = 1000,
  min_valid: int = 2,
  require_reference: bool = False,
) -> list[dict[str, object]]:
  """rank mi pairs by score and include original aln and ref pos

  all cols are numbered from 1 in output
  nan scores and pairs w too few obs are excluded
  ref gaps retain original aln cols and get empty ref pos in csv
  """

  _check_options(score, max_pairs, min_valid, require_reference)
  _check_block(block, x_map, y_map)

  return list(_ranked_pair_rows(
    block, x_map, y_map, score, max_pairs, min_valid, require_reference,
  ))


def _ranked_pair_rows(
  block: MIBlock | DCABlock,
  x_map: ReferenceCoordinates,
  y_map: ReferenceCoordinates,
  score: str,
  max_pairs: int | None,
  min_valid: int,
  require_reference: bool,
) -> Iterator[dict[str, object]]:
  spec = get_score(score)
  x_axis, y_axis = coordinate_axes((x_map, y_map), require_reference)
  values = spec.values(block)
  counts = block.n_valid
  if (len(x_axis), len(y_axis)) != values.shape:
    ix = np.ix_(x_axis, y_axis)
    values, counts = values[ix], counts[ix]
  if np.any(counts < min_valid):
    values = np.where(counts >= min_valid, values, np.nan)
  raw, corrected = (
    ("mi_bits", "mi_apc") if spec.method == "mi" else ("dca_frobenius", "dca_apc")
  )
  raw_values = get_score(raw).values(block)
  corrected_values = get_score(corrected).values(block)
  for flat, rank, competition_rank in ranked_indices(values, score, max_pairs):
    xi, yi = divmod(flat, len(y_axis))
    i, j = x_axis[xi], y_axis[yi]
    x, y = x_map.site(i), y_map.site(j)
    yield {
      "rank": rank,
      "competition_rank": competition_rank,
      "statistic": score,
      "direction": spec.direction,
      "score": float(values[xi, yi]),
      "pair": x_map.component + y_map.component,
      "x_component": x_map.component,
      "x_reference_id": x_map.reference_id,
      "x_col": x.current_column,
      "x_original_col": x.original_column,
      "x_reference_pos": x.reference_position,
      "x_symbol": x.reference_symbol,
      "y_component": y_map.component,
      "y_reference_id": y_map.reference_id,
      "y_col": y.current_column,
      "y_original_col": y.original_column,
      "y_reference_pos": y.reference_position,
      "y_symbol": y.reference_symbol,
      raw: float(raw_values[i, j]),
      corrected: float(corrected_values[i, j]),
      "n_valid": int(block.n_valid[i, j]),
      "valid_weight": float(block.valid_weight[i, j]),
    }


def write_mi_table(
  path: str | Path,
  block: MIBlock,
  x_map: ReferenceCoordinates,
  y_map: ReferenceCoordinates,
  *,
  score: str = "mi_apc",
  max_pairs: int | None = 1000,
  min_valid: int = 2,
  require_reference: bool = False,
) -> Path:
  """write ranked mi pairs to csv or tsv"""

  _check_options(score, max_pairs, min_valid, require_reference)
  _check_block(block, x_map, y_map)
  rows = _ranked_pair_rows(block, x_map, y_map, score, max_pairs, min_valid, require_reference)
  return _write_table(path, FIELDS, rows)


def write_dca_table(
  path: str | Path,
  pair: str,
  block: DCABlock,
  x_map: ReferenceCoordinates,
  y_map: ReferenceCoordinates,
  *,
  score: str = "dca_apc",
  max_pairs: int | None = 1000,
  min_valid: int = 2,
  require_reference: bool = False,
) -> Path:
  """Write a ranked DCA view after selecting the statistic on the full block."""
  _check_options(score, max_pairs, min_valid, require_reference, method="dca")
  if pair != x_map.component + y_map.component or pair not in {"AB", "AC", "BC"}:
    raise ValueError("DCA pair and reference maps must match")
  if not isinstance(block, DCABlock):
    raise TypeError("block must be a DCABlock")
  if any(arr.shape != (x_map.n_sites, y_map.n_sites) for arr in vars(block).values()):
    raise ValueError("DCA arrays must match reference maps")
  rows = _ranked_pair_rows(block, x_map, y_map, score, max_pairs, min_valid, require_reference)
  return _write_table(path, DCA_FIELDS, rows)


def _write_table(path, fields, rows):
  path = Path(path).expanduser()
  delimiter = {".csv": ",", ".tsv": "\t"}.get(path.suffix.lower())
  if delimiter is None:
    raise ValueError("output path must end in .csv or .tsv")
  path.parent.mkdir(parents=True, exist_ok=True)

  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
      handle,
      fieldnames=fields,
      delimiter=delimiter,
    )
    writer.writeheader()
    writer.writerows(rows)

  return path


def write_triplet_mi_tables(
  out_dir: str | Path,
  result: TripletMI,
  a_map: ReferenceCoordinates,
  b_map: ReferenceCoordinates,
  c_map: ReferenceCoordinates,
  *,
  score: str = "mi_apc",
  max_pairs: int | None = 1000,
  min_valid: int = 2,
  require_reference: bool = False,
  fmt: str = "tsv",
) -> dict[str, Path]:
  """export ab ac bc mi tables using corresponding ref maps"""

  if not isinstance(result, TripletMI):
    raise TypeError("result must be TripletMI")

  if fmt not in {"csv", "tsv"}:
    raise ValueError("fmt must be csv or tsv")

  _check_options(score, max_pairs, min_valid, require_reference)

  blocks = (
    ("ab", result.ab, a_map, b_map),
    ("ac", result.ac, a_map, c_map),
    ("bc", result.bc, b_map, c_map),
  )

  # check all comps before writing any output files
  for _, block, x_map, y_map in blocks:
    _check_block(block, x_map, y_map)

  paths = {}

  for name, block, x_map, y_map in blocks:
    path = Path(out_dir).expanduser() / f"mi_{name}.{fmt}"

    paths[name] = write_mi_table(
      path,
      block,
      x_map,
      y_map,
      score=score,
      max_pairs=max_pairs,
      min_valid=min_valid,
      require_reference=require_reference,
    )

  return paths
