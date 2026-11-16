
# tests for cli.py matched msa analysis and output files

import csv

import pytest
from typer.testing import CliRunner

from triplet_coevolution.cli import app


runner = CliRunner()


# write a minimal stockholm aln like jackhmmer output
@pytest.fixture
def inputs(tmp_path):
  a = tmp_path / "a.sto"
  a.write_text(
    "# STOCKHOLM 1.0\n"
    "a_ref A-CD\n"
    "a1 AACD\n"
    "a2 AACD\n"
    "a3 C-CD\n"
    "a4 C-CD\n"
    "//\n"
  )

  b = tmp_path / "b.fasta"
  b.write_text(
    ">b_ref\nAACC\n"
    ">b1\nAACC\n"
    ">b2\nAACC\n"
    ">b3\nCCAA\n"
    ">b4\nCCAA\n"
  )

  c = tmp_path / "c.fasta"
  c.write_text(
    ">c_ref\nACAC\n"
    ">c1\nACAC\n"
    ">c2\nACAC\n"
    ">c3\nCACA\n"
    ">c4\nCACA\n"
  )

  matches = tmp_path / "triplets.tsv"
  matches.write_text(
    "observation_id\ta_id\tb_id\tc_id\n"
    "obs2\ta2\tb2\tc2\n"
    "obs1\ta1\tb1\tc1\n"
    "obs3\ta3\tb3\tc3\n"
    "obs4\ta4\tb4\tc4\n"
  )

  out = tmp_path / "results"
  args = [
    "analyze",
    "--a", str(a),
    "--b", str(b),
    "--c", str(c),
    "--matches", str(matches),
    "--a-ref", "a_ref",
    "--b-ref", "b_ref",
    "--c-ref", "c_ref",
    "--out", str(out),
  ]
  return a, b, c, matches, out, args


def _rows(path, delimiter="\t"):
  with path.open(newline="", encoding="utf-8") as handle:
    return list(csv.DictReader(handle, delimiter=delimiter))


# registered subcommand should appear in cli help
def test_analyze_registered():
  result = runner.invoke(app, ["--help"])
  assert result.exit_code == 0
  assert "analyze" in result.output


# required inputs must be explicitly given
def test_analyze_help_requires_curated_inputs():
  result = runner.invoke(app, ["analyze", "--help"])
  assert result.exit_code == 0
  for name in ("--a", "--b", "--c", "--matches", "--a-ref", "--b-ref", "--c-ref"):
    assert name in result.output


# three protein msas and curated matches produce ab ac bc tables
def test_analyze_protein_triplet(inputs):
  _, _, _, _, out, args = inputs
  result = runner.invoke(app, args)

  assert result.exit_code == 0, result.output
  assert "matched obs: 4" in result.output
  assert "effective depth:" in result.output
  for name in ("ab", "ac", "bc"):
    rows = _rows(out / f"mi_{name}.tsv")
    assert rows
    assert {row["pair"] for row in rows} == {name.upper()}
    assert all(int(row["n_valid"]) >= 2 for row in rows)


# original msa cols must survive qc masking and ref numbering offsets
def test_masked_reference_positions(inputs):
  _, _, _, _, out, args = inputs
  result = runner.invoke(app, args + ["--min-coverage", "0.75", "--a-start", "101"])

  assert result.exit_code == 0, result.output
  assert "kept cols: A=3" in result.output
  rows = _rows(out / "mi_ab.tsv")
  assert all(row["x_original_col"] != "2" for row in rows)
  assert any(
    row["x_col"] == "2" and row["x_original_col"] == "3"
    and row["x_reference_pos"] == "102"
    for row in rows
  )


# can rank raw mi and export csv instead of tsv
def test_csv_raw_mi_and_limit(inputs):
  _, _, _, _, out, args = inputs
  result = runner.invoke(app, args + [
    "--score", "mi_bits", "--fmt", "csv", "--max-pairs", "1"
  ])

  assert result.exit_code == 0, result.output
  for name in ("ab", "ac", "bc"):
    rows = _rows(out / f"mi_{name}.csv", delimiter=",")
    assert len(rows) == 1
    assert float(rows[0]["mi_bits"]) >= 0


# reference gap cols can be excluded from exported tables
def test_reference_gap_filter(inputs):
  _, _, _, _, out, args = inputs
  result = runner.invoke(app, args + [
    "--require-reference", "--min-coverage", "0.5"
  ])

  assert result.exit_code == 0, result.output
  for row in _rows(out / "mi_ab.tsv"):
    assert row["x_reference_pos"] != ""
    assert row["y_reference_pos"] != ""


# plots are optional and create four figures when selected
def test_analyze_with_plots(inputs):
  _, _, _, _, out, args = inputs
  result = runner.invoke(app, args + ["--plots"])

  assert result.exit_code == 0, result.output
  for name in ("ab", "ac", "bc", "triplet"):
    path = out / "figures" / f"mi_apc_{name}.png"
    assert path.is_file()
    assert path.stat().st_size > 1000


# invalid ref name should fail before any output is written
def test_reject_unknown_ref(inputs):
  _, _, _, _, out, args = inputs
  args[args.index("--a-ref") + 1] = "unknown"
  result = runner.invoke(app, args)

  assert result.exit_code == 1
  assert "reference id" in result.output.lower()
  assert not out.exists()


# incomplete homolog mapping should fail rather than guess correspondences
def test_reject_unknown_match(inputs):
  _, _, _, table, out, args = inputs
  table.write_text(
    "observation_id\ta_id\tb_id\tc_id\n"
    "obs1\ta_missing\tb1\tc1\n"
  )
  result = runner.invoke(app, args)

  assert result.exit_code == 1
  assert "missing" in result.output.lower()
  assert not out.exists()


# absent input should show an error and not create output dir
def test_reject_missing_file(inputs):
  _, _, _, _, out, args = inputs
  args[args.index("--a") + 1] = str(out / "missing.sto")
  result = runner.invoke(app, args)

  assert result.exit_code == 1
  assert "not found" in result.output.lower()
  assert not out.exists()


# reject invalid scoring params before doing expensive analysis
@pytest.mark.parametrize("opt,value", [
  ("--score", "invalid"),
  ("--fmt", "json"),
  ("--min-valid", "0"),
  ("--max-pairs", "0"),
])
def test_reject_invalid_output_options(inputs, opt, value):
  _, _, _, _, out, args = inputs
  result = runner.invoke(app, args + [opt, value])

  assert result.exit_code == 1
  assert "error:" in result.output.lower()
  assert not out.exists()


# impossible coverage threshold should reject before exporting results
def test_reject_bad_coverage(inputs):
  _, _, _, _, out, args = inputs
  result = runner.invoke(app, args + ["--min-coverage", "1.5"])

  assert result.exit_code == 1
  assert "min_coverage" in result.output
  assert not out.exists()


# all cols removed should produce a clear error
def test_reject_empty_component_mask(inputs):
  a, _, _, _, out, args = inputs
  a.write_text("# STOCKHOLM 1.0\na_ref -\na1 -\na2 -\na3 -\na4 -\n//\n")
  result = runner.invoke(app, args)

  assert result.exit_code == 1
  assert "every col" in result.output
  assert not out.exists()


# nucleotide types are supported by the same cli
def test_mixed_alphabet_triplet(inputs):
  _, b, c, _, out, args = inputs
  b.write_text(
    ">b_ref\nACGT\n>b1\nACGT\n>b2\nACGT\n>b3\nTGCA\n>b4\nTGCA\n"
  )
  c.write_text(
    ">c_ref\nACGU\n>c1\nACGU\n>c2\nACGU\n>c3\nUGCA\n>c4\nUGCA\n"
  )
  result = runner.invoke(app, args + ["--b-type", "dna", "--c-type", "rna"])

  assert result.exit_code == 0, result.output
  assert (out / "mi_bc.tsv").exists()


# invalid molecular alphabet should be rejected
def test_reject_bad_type(inputs):
  _, _, _, _, out, args = inputs
  result = runner.invoke(app, args + ["--a-type", "unknown"])

  assert result.exit_code == 1
  assert "unknown" in result.output
  assert not out.exists()
