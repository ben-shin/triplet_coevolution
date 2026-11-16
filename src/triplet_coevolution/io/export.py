# export ranked mi pairs w original aln and ref positions

import csv
from heapq import nsmallest
from numbers import Integral
from pathlib import Path

import numpy as np

from triplet_coevolution.core.coordinates import ReferenceCoordinates
from triplet_coevolution.inference.mi import MIBlock, TripletMI


FIELDS = (
  "rank",
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


def _check_options(
  score: str,
  max_pairs: int | None,
  min_valid: int,
  require_reference: bool,
) -> None:
  """check ranking and export options"""

  if score not in {"mi_bits", "mi_apc"}:
    raise ValueError("score must be mi_bits or mi_apc")

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

  if block.n_valid.dtype.kind not in "iu" or np.any(block.n_valid < 0):
    raise ValueError("n_valid must contain nonnegative ints")

  if not np.isfinite(block.valid_weight).all() or np.any(block.valid_weight < 0):
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

  scores = getattr(block, score)

  # build candidates without storing full row dicts
  def candidates():
    for i in range(x_map.n_sites):
      if require_reference and x_map.reference_positions[i] is None:
        continue

      for j in range(y_map.n_sites):
        if require_reference and y_map.reference_positions[j] is None:
          continue

        value = float(scores[i, j])

        if np.isfinite(value) and block.n_valid[i, j] >= min_valid:
          yield (-value, i, j)

  # stable tie breaks use current aln cols
  if max_pairs is None:
    selected = sorted(candidates())
  else:
    selected = nsmallest(max_pairs, candidates())

  rows = []

  for rank, (_, i, j) in enumerate(selected, start=1):
    x = x_map.site(i)
    y = y_map.site(j)

    rows.append({
      "rank": rank,
      "pair": block.component_x + block.component_y,
      "x_component": block.component_x,
      "x_reference_id": x_map.reference_id,
      "x_col": x.current_column,
      "x_original_col": x.original_column,
      "x_reference_pos": x.reference_position,
      "x_symbol": x.reference_symbol,
      "y_component": block.component_y,
      "y_reference_id": y_map.reference_id,
      "y_col": y.current_column,
      "y_original_col": y.original_column,
      "y_reference_pos": y.reference_position,
      "y_symbol": y.reference_symbol,
      "mi_bits": float(block.mi_bits[i, j]),
      "mi_apc": float(block.mi_apc[i, j]),
      "n_valid": int(block.n_valid[i, j]),
      "valid_weight": float(block.valid_weight[i, j]),
    })

  return rows


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

  path = Path(path).expanduser()

  delimiter = {
    ".csv": ",",
    ".tsv": "\t",
  }.get(path.suffix.lower())

  if delimiter is None:
    raise ValueError("output path must end in .csv or .tsv")

  rows = ranked_mi_rows(
    block,
    x_map,
    y_map,
    score=score,
    max_pairs=max_pairs,
    min_valid=min_valid,
    require_reference=require_reference,
  )

  path.parent.mkdir(parents=True, exist_ok=True)

  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
      handle,
      fieldnames=FIELDS,
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
