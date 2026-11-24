"""Synthetic joint-law, diagnostics, native policy and bootstrap-refit checks."""

import json
from itertools import combinations, product

import numpy as np
import pytest

from triplet_coevolution.conditional_workflow import conditional_design
from triplet_coevolution.inference.gibbs_bootstrap import GibbsOptions, parse_sampler, prepare_gibbs
from triplet_coevolution.inference.nested import NestedOptions, compare_nested
from triplet_coevolution.inference.nested_solver import Layout, exact_law
from triplet_coevolution.validation.clade_splits import make_split_plan


def prepare(shape, strength, *, sweeps=32, n=2048):
    edges = (
        list(combinations(range(len(shape)), 2))
        if len(shape) <= 3
        else [(i, i + 1) for i in range(len(shape) - 1)]
    )
    layout = Layout(shape, edges, (), NestedOptions())
    rng = np.random.default_rng(24061)
    theta = np.zeros(layout.size)
    for term, arr in zip(layout.terms, layout.arrays(theta), strict=True):
        if strength is None:
            arr[:] = rng.normal(0, 0.15, size=arr.shape)
        elif len(term) == 2:
            arr[:] = [[strength, -strength], [-strength, strength]]
    theta = layout.projected(theta)
    settings = GibbsOptions(burn_sweeps=sweeps)
    sampler, diag = prepare_gibbs(
        layout, theta, settings, rows=n, draws=1, folds=1, rng=np.random.default_rng(24062)
    )
    return layout, theta, settings, sampler, diag


def test_conditionals_and_sweep_certificate_against_explicit_transition_matrix():
    dims = (2, 3, 2)
    layout, t, _, s, diag = prepare(dims, None, sweeps=2, n=1)
    law, _ = exact_law(layout, t)
    states = list(product(*(range(d) for d in dims)))
    x = np.array(states)
    kernel = np.eye(len(x))
    for i in range(3):
        conditional = s.probabilities(x, i)
        step = np.zeros_like(kernel)
        for k, state in enumerate(states):
            ctx = tuple(slice(None) if a == i else v for a, v in enumerate(state))
            np.testing.assert_allclose(conditional[k], law[ctx] / law[ctx].sum(), atol=4e-15)
            for j in range(dims[i]):
                nxt = list(state)
                nxt[i] = j
                step[k, states.index(tuple(nxt))] = conditional[k, j]
        kernel = kernel @ step
    np.testing.assert_allclose(law.ravel() @ kernel, law.ravel(), atol=5e-16)
    worst = 0.5 * abs(kernel @ kernel - law.ravel()).sum(axis=1).max()
    assert 0 < worst <= diag["single_endpoint_tv_bound"]


def test_beyond_enumeration_chain_and_intentionally_trapped_law():
    layout, t, policy, s, diag = prepare((2,) * 18, 0.22, n=4096)
    with pytest.raises(ValueError, match="max_joint_states"):
        exact_law(layout, t)
    assert diag["mixing_status"] == "passed"
    rows = s.endpoints(4096, policy, np.random.default_rng(8023))
    target = (1 + np.tanh(0.22)) / 2
    assert max(abs((rows[:, 1:] == rows[:, :-1]).mean(axis=0) - target)) < 4.5 * np.sqrt(
        target * (1 - target) / len(rows)
    )
    _, _, _, _, frozen = prepare((2, 2, 2), 17.0)
    assert frozen["mixing_status"] == "withheld" and not frozen["burn_certificate_pass"]
    assert "state[0]=0" in frozen["indicator_failures"] and len(frozen["pilot_seed_words"]) == 4


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"method": "gaussian"},
        {"method": "exact", "chains": 8},
        {"method": "gibbs", "burn_sweeps": True},
        {"method": "gibbs", "diagnostic_sweeps": 75},
        {"method": "gibbs", "max_rhat": 2},
        {"method": "gibbs", "min_ess": 9999},
        {"method": "gibbs", "max_total_tv": float("nan")},
        {"method": "gibbs", "local_i3": 1},
        {"method": "gibbs", "unknown": 3},
    ],
)
def test_strict_sampler_schema(value):
    with pytest.raises((ValueError, TypeError)):
        parse_sampler(value)


def test_cost_and_order_gates_before_sampling():
    options = GibbsOptions()
    for layout, rows, message in (
        (Layout((2, 2, 2), [], [(0, 1, 2)], NestedOptions()), 32, "pair-only"),
        (Layout((2,) * 15, [], (), NestedOptions()), 100_000_000, "work limit"),
        (Layout((2, 2, 2), [], (), NestedOptions(max_workspace_bytes=512)), 30, "memory limit"),
    ):
        with pytest.raises(ValueError, match=message):
            prepare_gibbs(
                layout, np.zeros(layout.size), options, rows=rows, draws=1, folds=1, rng=None
            )
    layout = Layout((2, 2, 2), [], (), NestedOptions())
    with pytest.raises(ValueError, match="zero-sum"):
        prepare_gibbs(layout, np.ones(layout.size), options, rows=3, draws=1, folds=1, rng=None)


def test_common_weights_masks_refits_local_tail_and_replay(monkeypatch):
    from test_nested_conditional import source

    from triplet_coevolution.core.alignment import ComponentMSA, MatchedTripletMSA
    from triplet_coevolution.inference import nested

    msa, maps, groups = source(80)
    seq = list(msa.b.sequences)
    seq[2] = seq[72] = "?"
    b = ComponentMSA("B", "dna", msa.b.ids, tuple(seq))
    msa = MatchedTripletMSA(msa.a, b, msa.c, msa.matches)
    plan = make_split_plan(groups, groups, test_groups=["g6", "g7"])
    mass = {key: (0 if i in (4, 70) else 1 + (i % 3) * 0.2) for i, key in enumerate(groups)}
    args = {
        "options": NestedOptions(l2=0.2, max_joint_states=2),
        "weight_options": {"policy": "supplied", "supplied_weights": mass},
        "triples": [(5, 5, 5)],
        "null_draws": 4,
        "seed": 41,
        "null_sampler": {"method": "gibbs", "local_i3": True},
    }
    fits = []
    original = nested.optimize

    def tracking(layout, train, weights):
        fits.append((layout.terms, train.copy(), weights.copy()))
        return original(layout, train, weights)

    monkeypatch.setattr(nested, "optimize", tracking)
    fold = compare_nested(msa, maps, plan, **args).fold_rows[0]
    out = fold["parametric_null"]
    assert out["status"] == "available" and out["valid_draws"] == 4
    assert fold["complete_training_rows"] == 58 and fold["complete_test_rows"] == 18
    assert (out["configurations"], out["p_floor"]) == (256, 0.2)
    assert (
        out["local_i3"][0]["sites"] == [["A", 5], ["B", 5], ["C", 5]]
        and out["local_i3"][0]["status"] == "available"
    )
    assert len(fits) == 10
    for first, second in zip(fits[::2], fits[1::2], strict=True):
        assert len(first[0]) + 1 == len(second[0])
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(first[2], fits[0][2])
        np.testing.assert_array_equal(first[2], second[2])
    assert not np.array_equal(fits[2][1], fits[4][1])
    assert out == compare_nested(msa, maps, plan, **args).fold_rows[0]["parametric_null"]
    exact = compare_nested(msa, maps, plan, **{**args, "null_sampler": {"method": "exact"}})
    assert exact.fold_rows[0]["parametric_null"]["status"] == "unavailable_exact_joint_state_budget"
    # One attempted refit fails on a synthetic draw. No valid-only tail is published.
    count = 0

    def fail_one(layout, train, weights):
        nonlocal count
        count += 1
        theta, diag = original(layout, train, weights)
        return theta, {**diag, "status": "failed"} if count == 3 else diag

    monkeypatch.setattr(nested, "optimize", fail_one)
    failed = compare_nested(msa, maps, plan, **args).fold_rows[0]["parametric_null"]
    assert (
        failed["invalid_draws"] == 1 and failed["p_conservative"] is failed["tail_wilson95"] is None
    )
    assert failed["local_i3"][0]["by_q"] is None


def test_native_conditional_command_and_in_run_policy(tmp_path):
    from pathlib import Path

    from test_report import demo_args
    from typer.testing import CliRunner

    from triplet_coevolution.cli import app

    base = Path(__file__).resolve().parents[1] / "examples" / "mixed"
    matches = tmp_path / "groups.tsv"
    lines = (base / "matches.tsv").read_text().splitlines()
    matches.write_text(
        "\n".join(
            [
                lines[0] + "\tclade",
                *(
                    line + "\t" + ("train" if i % 2 == 0 else "test")
                    for i, line in enumerate(lines[1:])
                ),
            ]
        )
        + "\n"
    )
    policy = {
        "clades": {
            "source": "synthetic fixed groups",
            "clade_column": "clade",
            "test_groups": ["test"],
            "weight_policy": "uniform",
        },
        "panel": {"A": [1], "B": [2], "C": [1]},
        "triples": [[1, 2, 1]],
        "options": {"l2": 0.3, "max_joint_states": 2},
        "null_draws": 2,
        "seed": 31,
        "null_sampler": {"method": "gibbs"},
    }
    cfg = tmp_path / "policy.json"
    cfg.write_text(json.dumps(policy))
    source = tmp_path / "scored"
    out = tmp_path / "compare"
    for args in (
        [*demo_args(source, matches=matches), "--conditional-policy", str(cfg)],
        ["conditional", "--run", str(source), "--config", str(cfg), "--out", str(out)],
    ):
        r = CliRunner().invoke(app, args, catch_exceptions=False)
        assert r.exit_code == 0, r.output
    read = lambda root: json.loads((root / "fit_1/nested_summary.json").read_text())["folds"][0][
        "parametric_null"
    ]
    assert read(out) == read(source / "conditional")
    assert read(out)["status"] == "available"
    withheld = {
        **policy,
        "null_sampler": {"method": "gibbs", "burn_sweeps": 1, "max_total_tv": 1e-15},
    }
    cfg.write_text(json.dumps(withheld))
    unused = tmp_path / "withheld"
    r = CliRunner().invoke(
        app,
        ["conditional", "--run", str(source), "--config", str(cfg), "--out", str(unused)],
        catch_exceptions=False,
    )
    assert r.exit_code == 0, r.output
    attempt = read(unused)
    assert attempt["diagnostics"]["burn_certificate_pass"] is False
    assert attempt["status"] == "withheld_gibbs_mixing" and attempt["draws"] == 0
    assert attempt["p_conservative"] is attempt["tail_wilson95"] is None
    with pytest.raises(ValueError, match="positive null_draws"):
        conditional_design({**policy, "null_draws": 0})
