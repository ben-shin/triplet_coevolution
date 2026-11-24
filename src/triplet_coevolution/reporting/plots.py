
# plot mi and apc matrices w ref pos and obs coverage

from numbers import Integral
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
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

  unknown_support = (
    (block.n_valid == -1) & np.isnan(block.mi_bits) & np.isnan(block.mi_apc)
    & np.isnan(block.valid_weight)
  )
  if block.n_valid.dtype.kind not in "iu" or np.any((block.n_valid < 0) & ~unknown_support):
    raise ValueError("n_valid must contain nonnegative ints or a verified unknown-support cell")

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

def _mi_cmap(score: str, palette: str = "classic") -> LinearSegmentedColormap:
  """Preserve the historical palette for direct plot calls and explicit legacy runs."""
  if palette != "classic":
    from triplet_coevolution.reporting.explore_figures import PALETTES
    if palette not in PALETTES:
      raise ValueError(f"unknown MI palette {palette!r}")
    return LinearSegmentedColormap.from_list(
      palette, PALETTES[palette][int(score == "mi_apc")], N=257
    ).with_extremes(bad="#696e78")

  if score == "mi_apc":
    colors = ["#06ff00", "#000000", "#ff2500"]
  else:
    colors = ["#000000", "#ff2500"]

  # keep zero black and missing obs dark gray
  return LinearSegmentedColormap.from_list(
    f"mi_{score}",
    colors,
    N=257,
  ).with_extremes(bad="#252525")

def _draw_block(
  ax: Axes,
  block: MIBlock,
  x_map: ReferenceCoordinates,
  y_map: ReferenceCoordinates,
  data: np.ndarray,
  score: str,
  max_ticks: int,
  limits: tuple[float, float],
  palette: str,
):
  """draw one mi matrix w ref labels and nan cells masked"""

  cmap = _mi_cmap(score,palette)

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
    ax.text(0.5, 0.5, "no valid scores", color="white", ha="center", va="center", transform=ax.transAxes)

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
  palette: str = "classic",
) -> tuple[Figure, Axes]:
  """plot one mi block using reference seq numbering"""

  _check_options(score, min_valid, max_ticks)
  if not isinstance(show_colorbar, bool):
    raise TypeError("show_colorbar must be boolean")

  data = _score_data(block, x_map, y_map, score, min_valid)
  limits = _color_limits([data], score)

  fig, ax = plt.subplots(figsize=figsize)
  image = _draw_block(ax, block, x_map, y_map, data, score, max_ticks, limits,palette)

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
  palette: str = "classic",
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
    images.append(_draw_block(ax, block, x_map, y_map, values, score, max_ticks, limits,palette))

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
  palette: str = "accessible",
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
            block,
            x_map,
            y_map,
            score=score,
            min_valid=min_valid,
            max_ticks=max_ticks,
            palette=palette,
        )
        path = out_dir / f"{score}_{name}.{fmt}"
        try:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        finally:
            plt.close(fig)
        saved[name] = path

    fig, _ = plot_triplet_mi(
        result,
        a_map,
        b_map,
        c_map,
        score=score,
        min_valid=min_valid,
        max_ticks=max_ticks,
        palette=palette,
    )
    path = out_dir / f"{score}_triplet.{fmt}"
    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    saved["triplet"] = path

    return saved


def save_dca_plot(
    out_dir: Path,
    blocks: dict,
    maps: tuple[ReferenceCoordinates, ...],
    *,
    score: str,
    min_valid: int,
    dca_design=None,
) -> Path:
    """Render all three DCA blocks on a shared scale in the selected statistic's units."""
    from triplet_coevolution.inference.scores import get_score

    spec = get_score(score, "dca")
    axes_by_name = {mapping.component: mapping for mapping in maps}
    matrices = []
    for pair in ("AB", "AC", "BC"):
        block = blocks[pair]
        values = np.where(block.n_valid >= min_valid, spec.values(block), np.nan)
        matrices.append(values)
    finite = [np.abs(values[np.isfinite(values)]) for values in matrices]
    largest = max((float(np.max(values)) for values in finite if values.size), default=1.0) or 1.0
    signed = spec.correction == "APC"
    cmap = plt.get_cmap("RdBu_r" if signed else "viridis").with_extremes(bad="#cecece")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    path = out_dir / f"{score}_triplet.png"
    try:
        for axis, pair, values in zip(axes, ("AB", "AC", "BC"), matrices, strict=True):
            x, y = (axes_by_name[name] for name in pair)
            image = axis.imshow(
                np.ma.masked_invalid(values),
                cmap=cmap,
                vmin=-largest if signed else 0,
                vmax=largest,
                aspect="auto",
                interpolation="nearest",
                origin="upper",
            )
            axis.set_xticks(*_axis_ticks(y, 6), rotation=90)
            axis.set_yticks(*_axis_ticks(x, 6))
            axis.set(
                xlabel=f"{pair[1]} · {y.reference_id}",
                ylabel=f"{pair[0]} · {x.reference_id}",
                title=f"{pair} {score}",
            )
        fig.colorbar(
            image,
            ax=axes,
            shrink=0.8,
            label=f"{score} ({spec.metadata(dca_design=dca_design)['units']})",
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


def save_triple_plot(directory: Path, table, limit: int = 12) -> Path:
    """Plot leading finite ABC candidates in their declared rank order."""
    from triplet_coevolution.inference.scores import get_score

    spec = get_score(table.score, "threebody")
    rows = table.first_rows(directory, limit)
    fig, axis = plt.subplots(figsize=(8, max(3, 1.2 + 0.37 * len(rows))))
    path = directory / "figures" / f"{spec.name}_abc.png"
    try:
        if rows:
            values = [float(row[spec.name]) for row in rows]
            labels = [
                " / ".join(
                    row[key + "_reference_pos"] or "col" + row[key + "_original_col"]
                    for key in ("a", "b", "c")
                )
                for row in rows
            ]
            axis.barh(
                range(len(rows)), values, color=["#396d90" if v < 0 else "#9a4a24" for v in values]
            )
            axis.set_yticks(range(len(rows)), labels)
            axis.invert_yaxis()
        else:
            axis.text(0.5, 0.5, "No finite candidate scores", transform=axis.transAxes, ha="center")
        axis.axvline(0, color="#49545a", linewidth=0.8)
        axis.set(
            xlabel=f"{spec.name} ({spec.units})",
            ylabel="A / B / C positions",
            title=f"Leading tested triples · {spec.direction} first",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)
    return path
