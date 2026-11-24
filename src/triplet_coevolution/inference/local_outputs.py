"""Publish local probability-model diagnostics without retaining probability tensors."""

import csv
import json
import math
from collections import Counter
from dataclasses import asdict

import numpy as np


def output_value(value):
    return None if isinstance(value, float) and not math.isfinite(value) else value


def write_local_fits(
    directory, msa, weights, selection, axes, settings, fmt, min_valid, min_valid_weight
):
    from triplet_coevolution.inference.maxent import MAXENT_FIELDS, fit_maxent_grid

    support = settings.support(msa)
    zero_bins = settings.zero_bins(selection.maps, support, axes)
    if zero_bins.keys() - set(selection.triples):
        raise ValueError("structural zero specified for an unselected ABC candidate")
    frozen = support.metadata()
    frozen["coordinates"] = {
        m.component: {
            "reference_id": m.reference_id,
            "original_columns": list(m.original_positions),
            "reference_positions": list(m.reference_positions),
        }
        for m in selection.maps
    }
    frozen["structural_zeros"] = [
        {"original_columns": list(orig), "bins": list(bins)}
        for orig, bins in settings.structural_zeros
    ]
    (directory / "local_state_support.json").write_text(
        json.dumps(frozen, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    primary = []
    status = Counter()
    information = Counter()
    table = directory / f"local_model_fits.{fmt}"
    coordinates = iter(selection.rows())
    count = 0
    with table.open("w", encoding="utf8", newline="") as stream:
        names = [
            "candidate_id",
            *(f"{c}_{k}" for c in "abc" for k in ("original_col", "reference_id", "reference_pos")),
            "setting",
            "prior_mode",
            "prior_mass",
            *MAXENT_FIELDS,
        ]
        writer = csv.DictWriter(
            stream, fieldnames=names, delimiter="\t" if fmt == "tsv" else ",", lineterminator="\n"
        )
        writer.writeheader()
        for candidates in selection.batches(32):
            masks = {}
            for triple in candidates:
                if triple in zero_bins:
                    mask = np.ones(support.shape_for(triple), dtype=bool)
                    for cell in zero_bins[triple]:
                        mask[cell] = False
                    masks[triple] = mask
            fits = fit_maxent_grid(
                msa,
                candidates,
                weights,
                support=support,
                options_sequence=settings.options,
                min_valid=min_valid,
                min_valid_weight=min_valid_weight,
                structural_support=masks,
            )
            for j, result in enumerate(fits[0]):
                coordinates_row = next(coordinates)
                key = {k: coordinates_row[k] for k in names[:10]}
                primary.append(result)
                status[result.status] += 1
                information[result.information_status] += 1
                for index, (option, results) in enumerate(zip(settings.options, fits, strict=True)):
                    row = results[j]
                    writer.writerow(
                        {
                            **key,
                            "setting": index,
                            "prior_mode": option.prior_mode,
                            "prior_mass": option.prior_mass,
                            **{k: output_value(getattr(row, k)) for k in MAXENT_FIELDS},
                        }
                    )
                    count += 1
    return primary, {
        "schema_version": 1,
        "model": "local M1/M2/M3",
        "ranked_setting": 0,
        "settings": [asdict(o) for o in settings.options],
        "include_gaps": settings.include_gaps,
        "min_state_count": settings.min_state_count,
        "primary_fit_status": dict(status),
        "primary_information_status": dict(information),
        "diagnostic_rows": count,
        "diagnostic_table": table.name,
        "frozen_support": "local_state_support.json",
        "calibration": "Descriptive local model comparison. Pair-preserving M2 calibration was not performed.",
    }
