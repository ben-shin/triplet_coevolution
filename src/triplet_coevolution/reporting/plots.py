
# plot mi and apc matrices w ref pos and obs coverage

from numbers import Integral
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from triplet_coevolution.core.coordinates import ReferenceCoordinates
from triplet_coevolution.inference.mi import MIBlock, TripletMI


def _check_options(score: str, min_valid: int, max_ticks: int) -> None:
  """validate score and plotting limits"""

  if score not in {"mi_bits", "mi_apc"}:
    raise ValueError("score must be mi_bits or mi_apc")

  for name, value in (("min_valid", min_valid), ("max_ticks", max_ticks)):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
      raise ValueError(f"{name} must be a positive int")


def _score_data(
  block: MIBlock,
  x_map: ReferenceCoordinates,
  y_map: ReferenceCoordinates,
  score: str,
  min_valid: int,
) -> np.ndarray:
  """filter mi scores using per pair obs support"""

  if not isinstance(block, MIBlock):
    raise TypeError("block must be an MIBlock")

  if not isinstance(x_map, ReferenceCoordinates) or not isinstance(y_map, ReferenceCoordinates):
    raise TypeError("maps must be ReferenceCoordinates")

  if (block.component_x, block.component_y) != (x_map.component, y_map.component):
    raise ValueError("mi comp labels dont match ref maps")

  shape = (x_map.n_sites, y_map.n_sites)

  for name in ("mi_bits", "mi_apc", "n_valid"):
    if np.asarray(getattr(block, name)).shape != shape:
      raise ValueError(f"{name} shape doesnt match ref maps")

  if block.n_valid.dtype.kind not in "iu" or np.any(block.n_valid < 0):
    raise ValueError("n_valid must contain nonnegative ints")

  data = np.asarray(getattr(block, score), dtype=float).copy()

  if np.isinf(data).any():
    raise ValueError("mi scores cant contain infinity")

  if score == "mi_bits" and np.any(np.isfinite(data) & (data < 0)):
    raise ValueError("raw mi cant be negative")

  data[block.n_valid < min_valid] = np.nan
  return data


def _color_limits(data: list[np.ndarray], score: str) -> tuple[float, float]:
  """use sequential scale for mi and symmetric scale for apc"""

  finite = [part[np.isfinite(part)] for part in data]
  finite = [part for part in finite if part.size]

  if not finite:
    return (0.0, 1.0) if score == "mi_bits" else (-1.0, 1.0)

  largest = max(float(np.max(np.abs(part))) for part in finite)
  largest = largest if largest > 0 else 1.0

  if score == "mi_bits":
    return 0.0, largest

  return -largest, largest


def _axis_ticks(mapping: ReferenceCoordinates, max_ticks: int) -> tuple[np.ndarray, list[str]]:
  """label a few aln cols using ref pos when available"""

  n = min(mapping.n_sites, max_ticks)
  indices = np.unique(np.linspace(0, mapping.n_sites - 1, n, dtype=int))

  labels = []
  for i in indices:
    pos = mapping.reference_positions[i]
    labels.append(str(pos) if pos is not None else f"col{mapping.original_positions[i]}")

  return indices, labels


def _draw_block(
  ax: Axes,
  block: MIBlock,
  x_map: ReferenceCoordinates,
  y_map: ReferenceCoordinates,
  data: np.ndarray,
  score: str,
  max_ticks: int,
  limits: tuple[float, float],
):
  """draw one mi matrix w ref labels and nan cells masked"""

  cmap = plt.get_cmap("viridis" if score == "mi_bits" else "RdBu_r").copy()
  cmap.set_bad("0.85")

  image = ax.imshow(
    np.ma.masked_invalid(data),
    aspect="auto",
    interpolation="nearest",
    origin="upper",
    cmap=cmap,
    vmin=limits[0],
    vmax=limits[1],
  )

  x_ticks, x_labels = _axis_ticks(y_map, max_ticks)
  y_ticks, y_labels = _axis_ticks(x_map, max_ticks)

  ax.set_xticks(x_ticks, x_labels, rotation=90)
  ax.set_yticks(y_ticks, y_labels)
  ax.set_xlabel(f"{block.component_y} ref pos ({y_map.reference_id})")
  ax.set_ylabel(f"{block.component_x} ref pos ({x_map.reference_id})")
  ax.set_title(f"{block.component_x}{block.component_y} {score}")

  if not np.isfinite(data).any():
    ax.text(0.5, 0.5, "no valid scores", ha="center", va="center", transform=ax.transAxes)

  return image


def plot_mi_block(
  block: MIBlock,
  x_map: ReferenceCoordinates,
  y_map: ReferenceCoordinates,
  *,
  score: str = "mi_apc",
  min_valid: int = 2,
  max_ticks: int = 8,
  show_colorbar: bool = True,
  figsize: tuple[float, float] = (8, 5),
) -> tuple[Figure, Axes]:
  """plot one mi block using reference seq numbering"""

  _check_options(score, min_valid, max_ticks)
  if not isinstance(show_colorbar, bool):
    raise TypeError("show_colorbar must be boolean")

  data = _score_data(block, x_map, y_map, score, min_valid)
  limits = _color_limits([data], score)

  fig, ax = plt.subplots(figsize=figsize)
  image = _draw_block(ax, block, x_map, y_map, data, score, max_ticks, limits)

  if show_colorbar and np.isfinite(data).any():
    fig.colorbar(image, ax=ax, label=f"{score} (bits)")

  fig.tight_layout()
  return fig, ax


def plot_triplet_mi(
  result: TripletMI,
  a_map: ReferenceCoordinates,
  b_map: ReferenceCoordinates,
  c_map: ReferenceCoordinates,
  *,
  score: str = "mi_apc",
  min_valid: int = 2,
  max_ticks: int = 6,
  shared_scale: bool = True,
  figsize: tuple[float, float] = (15, 4.5),
) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
  """plot ab ac bc together using matched score scales"""

  if not isinstance(result, TripletMI):
    raise TypeError("result must be TripletMI")

  _check_options(score, min_valid, max_ticks)
  if not isinstance(shared_scale, bool):
    raise TypeError("shared_scale must be boolean")

  blocks = (
    (result.ab, a_map, b_map),
    (result.ac, a_map, c_map),
    (result.bc, b_map, c_map),
  )

  data = [_score_data(block, x, y, score, min_valid) for block, x, y in blocks]
  common_limits = _color_limits(data, score)
  fig, axes = plt.subplots(1, 3, figsize=figsize)
  images = []

  for ax, (block, x_map, y_map), values in zip(axes, blocks, data):
    limits = common_limits if shared_scale else _color_limits([values], score)
    images.append(_draw_block(ax, block, x_map, y_map, values, score, max_ticks, limits))

  if shared_scale:
    if any(np.isfinite(values).any() for values in data):
      fig.colorbar(images[0], ax=axes, shrink=0.8, label=f"{score} (bits)")
  else:
    for ax, image, values in zip(axes, images, data):
      if np.isfinite(values).any():
        fig.colorbar(image, ax=ax, label=f"{score} (bits)")
    fig.tight_layout()

  return fig, tuple(axes)


def save_mi_plots(
  out_dir: str | Path,
  result: TripletMI,
  a_map: ReferenceCoordinates,
  b_map: ReferenceCoordinates,
  c_map: ReferenceCoordinates,
  *,
  score: str = "mi_apc",
  min_valid: int = 2,
  max_ticks: int = 8,
  fmt: str = "png",
  dpi: int = 200,
) -> dict[str, Path]:
  """save ab ac bc and combined heatmaps then close figs"""

  _check_options(score, min_valid, max_ticks)

  if fmt not in {"png", "pdf", "svg"}:
    raise ValueError("fmt must be png, pdf, or svg")

  if isinstance(dpi, bool) or not isinstance(dpi, Integral) or dpi < 1:
    raise ValueError("dpi must be a positive int")

  if not isinstance(result, TripletMI):
    raise TypeError("result must be TripletMI")

  blocks = (
    ("ab", result.ab, a_map, b_map),
    ("ac", result.ac, a_map, c_map),
    ("bc", result.bc, b_map, c_map),
  )

  # validate all score matrices before creating output files
  for _, block, x_map, y_map in blocks:
    _score_data(block, x_map, y_map, score, min_valid)

  out_dir = Path(out_dir).expanduser()
  out_dir.mkdir(parents=True, exist_ok=True)
  saved = {}

  for name, block, x_map, y_map in blocks:
    fig, _ = plot_mi_block(
      block, x_map, y_map, score=score, min_valid=min_valid, max_ticks=max_ticks,
    )
    path = out_dir / f"{score}_{name}.{fmt}"
    try:
      fig.savefig(path, dpi=dpi, bbox_inches="tight")
    finally:
      plt.close(fig)
    saved[name] = path

  fig, _ = plot_triplet_mi(
    result, a_map, b_map, c_map,
    score=score, min_valid=min_valid, max_ticks=max_ticks,
  )
  path = out_dir / f"{score}_triplet.{fmt}"
  try:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
  finally:
    plt.close(fig)
  saved["triplet"] = path

  return saved
