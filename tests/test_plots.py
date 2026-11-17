
# tests for reporting/plots.py mi heatmaps and ref labels

from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as to_hex
import numpy as np
import pytest

from triplet_coevolution.core.alignment import (
  ComponentMSA,
  MatchedTripletMSA,
  TripletMatch,
)
from triplet_coevolution.core.coordinates import build_reference_map, subset_reference_map
from triplet_coevolution.inference.mi import MIBlock, TripletMI, triplet_mi
from triplet_coevolution.reporting.plots import (
  plot_mi_block,
  plot_triplet_mi,
  save_mi_plots,
)


# maps include ref gaps and original aln pos after masking
@pytest.fixture
def ref_maps():
  a = ComponentMSA("A", "protein", ("a_ref",), ("A-CDE",))
  b = ComponentMSA("B", "dna", ("b_ref",), ("G-T",))
  c = ComponentMSA("C", "rna", ("c_ref",), ("AU",))

  a_map = subset_reference_map(build_reference_map(a, "a_ref", 101), (0, 1, 3, 4))
  b_map = build_reference_map(b, "b_ref", 31)
  c_map = build_reference_map(c, "c_ref", 51)
  return a_map, b_map, c_map


# fixed matrices w missing scores and variable obs support
@pytest.fixture
def ab_block():
  return MIBlock(
    "A", "B",
    np.array([
      [0.0, 0.2, 0.3],
      [0.5, np.nan, 0.1],
      [0.7, 0.8, 0.9],
      [0.0, 0.2, 0.1],
    ]),
    np.array([
      [-0.2, 0.1, 0.3],
      [0.5, np.nan, -0.1],
      [-0.4, 0.0, 0.6],
      [0.1, 0.2, -0.3],
    ]),
    np.array([
      [5, 5, 1],
      [5, 0, 5],
      [5, 5, 5],
      [5, 5, 5],
    ]),
    np.array([
      [3, 3, 0.5],
      [3, 0, 3],
      [3, 3, 3],
      [3, 3, 3],
    ]),
  )


# all three blocks use their own comp map dimensions
@pytest.fixture
def triplet_result(ab_block):
  ac = MIBlock(
    "A", "C",
    np.full((4, 2), 0.7),
    np.full((4, 2), 0.4),
    np.full((4, 2), 5, dtype=int),
    np.full((4, 2), 3.0),
  )
  bc = MIBlock(
    "B", "C",
    np.full((3, 2), 0.5),
    np.full((3, 2), -0.7),
    np.full((3, 2), 5, dtype=int),
    np.full((3, 2), 3.0),
  )
  return TripletMI(ab_block, ac, bc, 5, 3.0, False, 2)


# single block should display correctly mapped ref pos
@pytest.mark.parametrize("score", ["mi_bits", "mi_apc"])
def test_plot_block_labels(ab_block, ref_maps, score):
  a_map, b_map, _ = ref_maps
  fig, ax = plot_mi_block(ab_block, a_map, b_map, score=score)
  try:
    assert ax.images[0].get_array().shape == (4, 3)
    assert "B ref pos" in ax.get_xlabel()
    assert "A ref pos" in ax.get_ylabel()
    assert score in ax.get_title()
    assert "col2" in [tick.get_text() for tick in ax.get_xticklabels()]
    assert "col2" in [tick.get_text() for tick in ax.get_yticklabels()]
    assert "101" in [tick.get_text() for tick in ax.get_yticklabels()]
  finally:
    plt.close(fig)


# low obs support and undefined scores should be masked
@pytest.mark.parametrize("score", ["mi_bits", "mi_apc"])
def test_mask_missing_scores(ab_block, ref_maps, score):
  a_map, b_map, _ = ref_maps
  fig, ax = plot_mi_block(ab_block, a_map, b_map, score=score)
  try:
    data = ax.images[0].get_array()
    assert np.ma.is_masked(data[0, 2])
    assert np.ma.is_masked(data[1, 1])
    assert not np.ma.is_masked(data[2, 2])
  finally:
    plt.close(fig)


# apc uses symmetric limits to keep zero in center
def test_apc_color_scale(ab_block, ref_maps):
  a_map, b_map, _ = ref_maps
  fig, ax = plot_mi_block(ab_block, a_map, b_map)
  try:
    assert ax.images[0].norm.vmin == pytest.approx(-0.6)
    assert ax.images[0].norm.vmax == pytest.approx(0.6)
  finally:
    plt.close(fig)


# raw mi starts at zero and uses sequential intensity
def test_raw_mi_color_scale(ab_block, ref_maps):
  a_map, b_map, _ = ref_maps
  fig, ax = plot_mi_block(ab_block, a_map, b_map, score="mi_bits")
  try:
    assert ax.images[0].norm.vmin == pytest.approx(0)
    assert ax.images[0].norm.vmax == pytest.approx(0.9)
  finally:
    plt.close(fig)


# plot may be used without a colorbar
@pytest.mark.parametrize("show_colorbar,expected", [(True, 2), (False, 1)])
def test_colorbar_option(ab_block, ref_maps, show_colorbar, expected):
  a_map, b_map, _ = ref_maps
  fig, _ = plot_mi_block(ab_block, a_map, b_map, show_colorbar=show_colorbar)
  try:
    assert len(fig.axes) == expected
  finally:
    plt.close(fig)


# no usable obs should give a clearly empty heatmap
def test_all_invalid_scores(ab_block, ref_maps):
  a_map, b_map, _ = ref_maps
  empty = replace(ab_block, n_valid=np.zeros((4, 3), dtype=int))
  fig, ax = plot_mi_block(empty, a_map, b_map)
  try:
    assert np.ma.getmaskarray(ax.images[0].get_array()).all()
    assert any(x.get_text() == "no valid scores" for x in ax.texts)
    assert len(fig.axes) == 1
  finally:
    plt.close(fig)

# check apc palette uses green neg black zero and red pos
def test_classic_apc_colors(ab_block, ref_maps):
  a_map, b_map, _ = ref_maps
  fig, ax = plot_mi_block(ab_block, a_map, b_map)

  try:
    image = ax.images[0]
    cmap = image.get_cmap()
    norm = image.norm

    assert to_hex(cmap(norm(float(norm.vmin)))) == "#06ff00"
    assert to_hex(cmap(norm(0.0))) == "#000000"
    assert to_hex(cmap(norm(float(norm.vmax)))) == "#ff2500"

  finally:
    plt.close(fig)


# raw mi has no neg scores so use black to red
def test_classic_raw_mi_colors(ab_block, ref_maps):
  a_map, b_map, _ = ref_maps
  fig, ax = plot_mi_block(
    ab_block,
    a_map,
    b_map,
    score="mi_bits",
  )

  try:
    image = ax.images[0]
    cmap = image.get_cmap()
    norm = image.norm

    assert to_hex(cmap(norm(0.0))) == "#000000"
    assert to_hex(cmap(norm(float(norm.vmax)))) == "#ff2500"

  finally:
    plt.close(fig)


# undefined scores shouldnt look identical to zero
def test_classic_missing_color(ab_block, ref_maps):
  a_map, b_map, _ = ref_maps
  fig, ax = plot_mi_block(ab_block, a_map, b_map)

  try:
    image = ax.images[0]
    cmap = image.get_cmap()

    assert to_hex(cmap.get_bad()) == "#252525"
    assert to_hex(cmap(image.norm(0.0))) == "#000000"
    assert np.ma.is_masked(image.get_array()[1, 1])

  finally:
    plt.close(fig)


# the three panels must share one scale for score comparison
def test_triplet_shared_scale(triplet_result, ref_maps):
  fig, axes = plot_triplet_mi(triplet_result, *ref_maps)
  try:
    assert len(axes) == 3
    assert len(fig.axes) == 4
    assert [a.images[0].get_array().shape for a in axes] == [
      (4, 3), (4, 2), (3, 2),
    ]
    for ax in axes:
      assert ax.images[0].norm.vmin == pytest.approx(-0.7)
      assert ax.images[0].norm.vmax == pytest.approx(0.7)
  finally:
    plt.close(fig)


# optional independent scale gives one colorbar per comp pair
def test_triplet_separate_scales(triplet_result, ref_maps):
  fig, axes = plot_triplet_mi(triplet_result, *ref_maps, shared_scale=False)
  try:
    assert len(fig.axes) == 6
    assert axes[0].images[0].norm.vmax == pytest.approx(0.6)
    assert axes[1].images[0].norm.vmax == pytest.approx(0.4)
    assert axes[2].images[0].norm.vmax == pytest.approx(0.7)
  finally:
    plt.close(fig)


# save three blocks and combined matrix without leaking figs
def test_save_png_plots(tmp_path, triplet_result, ref_maps):
  initial = set(plt.get_fignums())
  paths = save_mi_plots(tmp_path / "figs", triplet_result, *ref_maps)

  assert set(paths) == {"ab", "ac", "bc", "triplet"}
  assert all(
    path.suffix == ".png" and path.stat().st_size > 1000
    for path in paths.values()
  )
  assert set(plt.get_fignums()) == initial


# vector output should also be supported
@pytest.mark.parametrize("fmt,header", [("pdf", b"%PDF"), ("svg", b"<?xml")])
def test_save_vector_plots(tmp_path, triplet_result, ref_maps, fmt, header):
  paths = save_mi_plots(tmp_path, triplet_result, *ref_maps, fmt=fmt)
  assert all(path.read_bytes().startswith(header) for path in paths.values())


# plotting shouldnt alter underlying scores
def test_no_input_mutation(ab_block, ref_maps):
  a_map, b_map, _ = ref_maps
  before = ab_block.mi_apc.copy()

  fig, _ = plot_mi_block(ab_block, a_map, b_map)
  plt.close(fig)

  np.testing.assert_allclose(ab_block.mi_apc, before, equal_nan=True)


# block must match the map lengths and labels
@pytest.mark.parametrize("name", ["mi_bits", "mi_apc", "n_valid"])
def test_wrong_score_shapes(ab_block, ref_maps, name):
  a_map, b_map, _ = ref_maps
  block = replace(ab_block, **{name: np.zeros((1, 2))})

  with pytest.raises(ValueError, match="shape"):
    plot_mi_block(block, a_map, b_map)


def test_wrong_ref_labels(ab_block, ref_maps):
  a_map, _, c_map = ref_maps
  with pytest.raises(ValueError, match="labels"):
    plot_mi_block(ab_block, a_map, c_map)


# reject invalid params before plotting anything
@pytest.mark.parametrize(
  "field,value,error",
  [
    ("score", "wrong", ValueError),
    ("min_valid", 0, ValueError),
    ("min_valid", True, ValueError),
    ("max_ticks", 0, ValueError),
    ("max_ticks", 2.5, ValueError),
    ("show_colorbar", "yes", TypeError),
  ],
)
def test_invalid_block_options(ab_block, ref_maps, field, value, error):
  a_map, b_map, _ = ref_maps
  with pytest.raises(error):
    plot_mi_block(ab_block, a_map, b_map, **{field: value})


# invalid matrix values shouldnt be plotted
@pytest.mark.parametrize("value", [np.inf, -0.2])
def test_invalid_raw_mi(ab_block, ref_maps, value):
  a_map, b_map, _ = ref_maps
  data = ab_block.mi_bits.copy()
  data[0, 0] = value

  with pytest.raises(ValueError):
    plot_mi_block(
      replace(ab_block, mi_bits=data),
      a_map,
      b_map,
      score="mi_bits",
    )


# invalid n_valid should fail early
def test_invalid_n_valid(ab_block, ref_maps):
  a_map, b_map, _ = ref_maps
  with pytest.raises(ValueError, match="n_valid"):
    plot_mi_block(
      replace(ab_block, n_valid=np.full((4, 3), -1)),
      a_map,
      b_map,
    )


# triplet plot must be given a triplet result
def test_invalid_triplet_result(ref_maps):
  with pytest.raises(TypeError, match="TripletMI"):
    plot_triplet_mi("not mi", *ref_maps)


# wrong format and dpi should fail before files are written
@pytest.mark.parametrize("fmt", ["jpg", "CSV", "txt"])
def test_invalid_plot_format(tmp_path, triplet_result, ref_maps, fmt):
  with pytest.raises(ValueError, match="fmt"):
    save_mi_plots(tmp_path, triplet_result, *ref_maps, fmt=fmt)

  assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("dpi", [0, True, 2.5, -1])
def test_invalid_dpi(tmp_path, triplet_result, ref_maps, dpi):
  with pytest.raises(ValueError, match="dpi"):
    save_mi_plots(tmp_path, triplet_result, *ref_maps, dpi=dpi)

  assert not list(tmp_path.iterdir())


# wrong block should prevent any file creation
def test_save_preflight(tmp_path, triplet_result, ref_maps):
  a_map, b_map, _ = ref_maps

  with pytest.raises(ValueError, match="labels"):
    save_mi_plots(
      tmp_path / "figs",
      triplet_result,
      a_map,
      b_map,
      b_map,
    )

  assert not (tmp_path / "figs").exists()


# actual inferred mi should work w the plotting interface
def test_real_mi_plot():
  a = ComponentMSA(
    "A", "protein",
    ("a1", "a2", "a3", "a4"),
    ("A", "A", "C", "C"),
  )
  b = ComponentMSA(
    "B", "dna",
    ("b1", "b2", "b3", "b4"),
    ("G", "G", "T", "T"),
  )
  c = ComponentMSA(
    "C", "rna",
    ("c1", "c2", "c3", "c4"),
    ("A", "C", "A", "C"),
  )

  matches = tuple(
    TripletMatch(f"obs{i}", f"a{i}", f"b{i}", f"c{i}")
    for i in range(1, 5)
  )

  results = triplet_mi(MatchedTripletMSA(a, b, c, matches))
  maps = (
    build_reference_map(a, "a1"),
    build_reference_map(b, "b1"),
    build_reference_map(c, "c1"),
  )

  fig, axes = plot_triplet_mi(results, *maps, score="mi_bits")
  try:
    assert axes[0].images[0].get_array()[0, 0] == pytest.approx(1)
  finally:
    plt.close(fig)
