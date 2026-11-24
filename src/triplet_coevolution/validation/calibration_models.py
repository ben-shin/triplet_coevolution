"""Bind an observed archive to exactly the requested score refits and coordinate axes."""

from itertools import product
from math import prod

import numpy as np

from triplet_coevolution.inference.candidates import candidate_id
from triplet_coevolution.inference.dca_models import DCADesign
from triplet_coevolution.inference.global_potts import global_triples
from triplet_coevolution.inference.maxent import fit_maxent_triples
from triplet_coevolution.inference.mi import triplet_mi
from triplet_coevolution.inference.potts import PAIRS, direct_couplings
from triplet_coevolution.inference.scores import (
    coordinate_axes,
    dca_with_apc_policy,
    get_score,
    mi_with_apc_policy,
)
from triplet_coevolution.inference.threebody import fit_triples
from triplet_coevolution.inference.weights import sequence_weights, weighting_preflight
from triplet_coevolution.validation.null_models import matching_null, prepare_m2_bootstrap
from triplet_coevolution.validation.null_policy import GENERATIVE
from triplet_coevolution.validation.pair_sampler import fit_pair_panel


class CalibrationModel:
    def __init__(self, data, policy, *, tree_inputs=None):
        self.data, self.policy = data, policy
        self.settings = data.store.provenance.get("methods", {})
        self.apc = data.store.provenance["apc_universe"]
        self.specs = [get_score(s) for s in policy.statistics]
        self.method = self.settings.get("threebody_model", "local")
        self.support = self.options = None
        self.masks = {}
        self.triples = ()
        self.candidate_scope = {}
        self.support_kwargs = {
            "min_valid": data.store.min_valid,
            "min_valid_weight": data.store.min_valid_weight,
        }
        self.diagnostics = {}
        self.axes = {}
        self.regions = {}
        self.groups = {}
        for spec in self.specs:
            spec.metadata(threebody_model=self.method)
            if spec.method == "threebody":
                if not self.triples:
                    self.triples, self.candidate_scope = data.triples()
                    if self.method == "maxent":
                        self.support, self.options, self.masks = data.local_settings(self.triples)
                self.regions[spec.name] = ["ABC"]
            else:
                if spec.name not in data.store.scores:
                    raise ValueError(f"statistic {spec.name} absent from archive")
                self.axes[spec.name] = coordinate_axes(
                    data.store.maps,
                    data.store.require_reference
                    or (spec.correction == "APC" and self.apc == "reference"),
                )
                self.regions[spec.name] = ["AB", "AC", "BC"]
            for region in self.regions[spec.name]:
                group = spec.name if policy.family == "statistic" else f"{spec.name}.{region}"
                self.groups.setdefault(group, []).append((spec.name, region))
        self.ids = [candidate_id(data.store.maps, t) for t in self.triples]
        self.worker_weight_budget = None
        self.tree_inputs = (
            tree_inputs
            if tree_inputs is not None
            else policy.tree.load(data.weights.observation_ids)
            if policy.tree
            else None
        )
        self._check_budget()
        if self.support and set(self.support.training_observations).intersection(
            data.weights.observation_ids
        ):
            raise ValueError(
                "calibration needs evaluation observations independent of training-derived state pooling"
            )
        if policy.null == "independent_site_tree":
            if self.masks:
                raise ValueError(
                    "independent-site simulation cannot preserve candidate-specific forbidden joint states"
                )
            self.tree, self.diagnostics = policy.tree.fit(
                data.msa, data.weights, loaded=self.tree_inputs
            )
            self.observed = self._project(
                self._matching_scores(data.msa, data.weights, archived=True)
            )
        elif policy.null in GENERATIVE:
            self._generative()
        else:
            self._matching()

    def sizes(self):
        def regionsize(name, region):
            if region == "ABC":
                return len(self.triples)
            x, y = ("ABC".index(c) for c in region)
            return len(self.axes[name][x]) * len(self.axes[name][y])

        return {g: sum(regionsize(n, r) for n, r in members) for g, members in self.groups.items()}

    def _check_budget(self):
        from triplet_coevolution.validation.null_runner import null_workspace

        p = self.policy
        sizes = self.sizes()
        if not all(sizes.values()):
            raise ValueError("no eligible coordinates in a declared test family")
        engine = null_workspace(
            sizes.values(),
            draws=p.draws,
            planned_draws=p.planned_draws,
            workers=p.workers,
            max_t=p.max_t,
        )
        rows = self.data.msa.n_observations
        widths = [c.n_sites for c in self.data.msa.components]
        cart = sum(widths[x] * widths[y] for x, y in PAIRS)
        peak = rows * sum(widths) * 8
        if self.tree_inputs is not None:
            from triplet_coevolution.validation.tree_model import tree_workspace

            trees = self.tree_inputs[0]
            trees = trees.values() if isinstance(trees, dict) else [trees]
            peak += tree_workspace(max(len(t.parents) for t in trees), rows, widths)
        if p.null not in GENERATIVE and self.data.weights.policy not in {
            "uniform",
            "clade_balanced",
            "supplied",
        }:
            self.worker_weight_budget = max(
                1, (p.memory_mib * 1024**2 - engine["runner_workspace_bytes"]) // p.workers
            )
            wp = weighting_preflight(
                self.data.msa,
                policy=self.data.weights.policy,
                component=self.data.weights.component,
                memory_budget_bytes=self.worker_weight_budget,
            )
            peak += wp["peak_bytes"]
        source_plan = self.data.root / "plan.json"
        if source_plan.is_file():
            from triplet_coevolution.validation.null_archive import read_json

            resources = read_json(self.data.artifact("plan.json")).get("estimates", {})
            if self.triples and self.method == "global":
                peak += resources.get("numeric_workspace_bytes", 0)
        else:
            resources = {}
        if any(s.method == "mi" for s in self.specs):
            peak += max(cart * 96, resources.get("mi_workspace_bytes", 0))
        if any(s.method == "dca" for s in self.specs):
            planned = resources.get("dca_workspace_bytes", 0)
            if design := self.settings.get("dca_design"):
                planned = max(
                    planned,
                    DCADesign.parse(design).preflight(self.data.msa)["estimated_numeric_bytes"],
                )
            peak += max(cart * 128, planned)
        if self.support:
            cell_counts = [prod(self.support.shape_for(t)) for t in self.triples]
            peak += max(cell_counts, default=0) * 128
            if p.null == "m2_bootstrap":
                peak += sum(c * 16 + rows * 32 for c in cell_counts) * len(self.specs)
        if p.null == "pair_panel" and self.support:
            joint = prod(max(m) + 1 for sites in self.support.maps for m in sites)
            if sum(widths) > 8 or joint > min(65536, self.options.max_cells):
                raise ValueError(
                    "exact pair_panel requires the entire retained panel to have 3–8 sites and at most 65,536 permitted configurations within max_cells"
                )
            peak += joint * 256
        worker_peak = peak * p.workers
        self.workspace = {"estimated_refit_bytes_all_workers": worker_peak, "runner": engine}
        self.runner_budget = p.memory_mib * 1024**2 - worker_peak
        if p.checkpoint and engine["estimated_checkpoint_bytes"] > p.checkpoint_mib * 1024**2:
            raise ValueError("checkpoint estimate exceeds checkpoint_mib")
        if self.runner_budget < engine["runner_workspace_bytes"]:
            raise ValueError(
                "calibration refit workspace exceeds memory_mib; use fewer workers or a smaller prespecified panel"
            )

    def _project(self, raw):
        return {g: np.concatenate([raw[k] for k in keys]) for g, keys in self.groups.items()}

    def _matching(self):
        d, p = self.data, self.policy
        if self.masks:
            raise ValueError(
                "matching shuffles may create forbidden joint states; use m2_bootstrap with the declared structural support"
            )
        strata = None
        if p.strata_column:
            if any(not row.get(p.strata_column) for row in d.metadata.values()):
                raise ValueError(f"missing stratum values for {p.strata_column!r}")
            strata = {k: r[p.strata_column] for k, r in d.metadata.items()}
        self.shuffle = matching_null(d.msa, p.null, strata=strata)
        self.diagnostics = self.shuffle.metadata()
        if p.null.startswith("preserve_"):
            self.diagnostics["conditioned_interface"] = p.null[-2:].upper()
            self.diagnostics["interpretation"] = (
                "The named pair remains intact. Its own statistic can change through joint conditioning, APC or reweighted observations. This control tests matching to the third component."
            )
        if d.weights.policy == "clade_balanced" and any(
            len({d.weights.clades[i] for i in group}) > 1 for group in self.shuffle.groups
        ):
            raise ValueError(
                "clade-balanced weights require shuffle strata nested within weight clades"
            )
        self.observed = self._project(self._matching_scores(d.msa, d.weights, archived=True))

    def _matching_scores(self, msa, weights, *, archived=False):
        raw = {}
        requested = {s.method for s in self.specs}
        prov = self.data.store.provenance
        pair = {}
        if not archived:
            if "mi" in requested:
                mi = mi_with_apc_policy(
                    triplet_mi(
                        msa,
                        weights,
                        kernel=prov["mi_kernel"],
                        include_gaps=prov.get("include_gaps", False),
                        **self.support_kwargs,
                    ),
                    self.data.store.maps,
                    self.apc,
                )
                pair["mi"] = {s: getattr(mi, s.lower()) for s in ("AB", "AC", "BC")}
            if "dca" in requested:
                pair["dca"] = dca_with_apc_policy(
                    DCADesign.parse(
                        self.settings.get("dca_design"), ridge=self.settings.get("ridge")
                    ).fit(msa, weights, mf=direct_couplings, **self.support_kwargs),
                    self.data.store.maps,
                    self.apc,
                )
        if "threebody" in requested:
            if self.method == "maxent":
                triples = fit_maxent_triples(
                    msa,
                    self.triples,
                    weights,
                    support=self.support,
                    options=self.options,
                    structural_support=self.masks,
                    **self.support_kwargs,
                )
            elif self.method == "global":
                try:
                    triples, _ = global_triples(
                        msa,
                        self.triples,
                        weights,
                        l2=self.settings["global_l2"],
                        **self.support_kwargs,
                    )
                except ValueError as exc:
                    if archived or not str(exc).startswith(
                        ("global Potts fit failed", "global Potts fit did not converge")
                    ):
                        raise
                    triples = [None] * len(self.triples)
            else:
                triples = fit_triples(
                    msa,
                    self.triples,
                    weights,
                    pseudocount=self.settings["pseudocount"],
                    **self.support_kwargs,
                )
        for spec in self.specs:
            if spec.method == "threebody":
                raw[spec.name, "ABC"] = np.array([getattr(t, spec.name, np.nan) for t in triples])
            else:
                axes = self.axes[spec.name]
                for i, j in PAIRS:
                    region = "ABC"[i] + "ABC"[j]
                    matrix = (
                        self.data.store.scores[spec.name][region]
                        if archived
                        else spec.values(pair[spec.method][region])
                    )
                    raw[spec.name, region] = matrix[np.ix_(axes[i], axes[j])].ravel()
        return raw

    def _generative(self):
        d, p = self.data, self.policy
        if any(s.method != "threebody" for s in self.specs) or self.method != "maxent":
            raise ValueError(
                "generative calibration requires a local maxent fit and an explicitly chosen local ABC statistic"
            )
        if d.weights.policy != {"uniform": "uniform", "fixed_design": "supplied"}[p.weight_design]:
            raise ValueError(
                "iid calibration requires --weight-policy uniform or an explicitly independent supplied design with weight_design=fixed_design"
            )
        self.local_generators = {}
        raw = {}
        if p.null == "m2_bootstrap":
            for spec in self.specs:
                models = [
                    prepare_m2_bootstrap(
                        d.msa,
                        t,
                        support=self.support,
                        options=self.options,
                        weights=d.weights,
                        weight_policy=p.weight_design,
                        selection=(
                            "uniform"
                            if self.candidate_scope.get("mode") == "uniform"
                            else "independent"
                            if self.candidate_scope.get("used_external_prior")
                            else "prespecified"
                        ),
                        score=spec.name,
                        allowed=self.masks.get(t),
                        **self.support_kwargs,
                    )
                    for t in self.triples
                ]
                self.local_generators[spec.name] = models
                self.diagnostics[spec.name] = [
                    {"candidate_id": key, **m.metadata()}
                    for key, m in zip(self.ids, models, strict=True)
                ]
                raw[spec.name, "ABC"] = np.array([m.observed for m in models])
        else:
            if self.masks:
                raise ValueError(
                    "exact pair_panel does not implement candidate-specific structural-zero constraints"
                )
            columns = []
            shape = []
            for component, site_maps in zip(d.msa.components, self.support.maps, strict=True):
                for index, lookup in enumerate(site_maps):
                    v = component.matrix[:, index]
                    keep = (v >= 0) & (v < len(lookup))
                    col = np.full(v.shape, -1, dtype=int)
                    col[keep] = np.asarray(lookup)[v[keep]]
                    columns.append(col)
                    shape.append(max(lookup) + 1)
            rows = np.column_stack(columns)
            self.panel = fit_pair_panel(
                rows,
                shape,
                weights=d.weights,
                options=self.options,
                weight_policy=p.weight_design,
                **self.support_kwargs,
            )
            self.diagnostics = self.panel.metadata()
            a, b, _ = (c.n_sites for c in d.msa.components)
            self.global_sites = [(i, a + j, a + b + k) for i, j, k in self.triples]
            for spec in self.specs:
                raw[spec.name, "ABC"] = np.array(
                    [self.panel.local_score(rows, t, score=spec.name) for t in self.global_sites]
                )
        self.observed = self._project(raw)

    def evaluate(self, replicate_id, streams):
        p = self.policy
        if p.null == "m2_bootstrap":
            raw = {
                (name, "ABC"): np.array(
                    [
                        m.draw(streams.generator("candidate", key))
                        for key, m in zip(self.ids, models, strict=True)
                    ]
                )
                for name, models in self.local_generators.items()
            }
        elif p.null == "pair_panel":
            if self.panel.status != "ok":
                return {k: np.full(v.shape, np.nan) for k, v in self.observed.items()}
            rows, panel = self.panel.draw_refit(streams.generator("pair_panel"))
            raw = {
                (spec.name, "ABC"): np.array(
                    [panel.local_score(rows, t, score=spec.name) for t in self.global_sites]
                )
                for spec in self.specs
            }
        else:
            shuffled = (
                self.tree.draw(streams.generator("independent_site_tree"))
                if p.null == "independent_site_tree"
                else self.shuffle.draw(streams.generator("matching"))
            )
            w = self.data.weights
            if w.policy not in {"uniform", "supplied", "clade_balanced"}:
                w = sequence_weights(
                    shuffled,
                    w.identity_threshold,
                    w.min_overlap,
                    policy=w.policy,
                    component=w.component,
                    memory_budget_bytes=self.worker_weight_budget,
                )
            raw = self._matching_scores(shuffled, w)
        return self._project(raw)

    def coordinate_rows(self, group):
        for name, region in self.groups[group]:
            roles = ["ABC".index(c) for c in region]
            coords = (
                self.triples if region == "ABC" else product(*(self.axes[name][r] for r in roles))
            )
            for j, indices in enumerate(coords):
                row = {
                    "statistic": name,
                    "interface": region,
                    "candidate_id": self.ids[j] if region == "ABC" else "",
                }
                for r, i in zip(roles, indices, strict=True):
                    m = self.data.store.maps[r]
                    site = m.site(i)
                    prefix = m.component.lower()
                    row[f"{prefix}_reference_id"] = m.reference_id
                    row[f"{prefix}_original_col"] = site.original_column
                    row[f"{prefix}_reference_pos"] = site.reference_position
                yield row
