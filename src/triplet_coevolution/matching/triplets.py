"""Construct and audit one-to-one triplets from explicit evidence groups."""

from triplet_coevolution.config import COMPONENTS
from triplet_coevolution.core.alignment import TripletMatch
from triplet_coevolution.matching.paralogs import select_candidate


def match_groups(available, groups, policy):
    assigned = {name: set() for name in COMPONENTS}
    matches = []
    audit = []
    for group in sorted(groups):
        selected = {}
        details = {}
        problems = []
        for name in COMPONENTS:
            declared = sorted(
                (c for c in groups[group] if c.component == name), key=lambda c: c.sequence_id
            )
            assigned[name].update(c.sequence_id for c in declared)
            candidates = [c for c in declared if c.sequence_id in available[name]]
            chosen = select_candidate(candidates, policy)
            details[name] = {
                "candidates": [
                    {
                        "id": c.sequence_id,
                        "priority": c.priority,
                        "available": c.sequence_id in available[name],
                    }
                    for c in declared
                ],
                "selected": chosen,
            }
            if chosen is None:
                problems.append(f"{name}: {'ambiguous' if candidates else 'missing'}")
            else:
                selected[name] = chosen
        if not problems:
            matches.append(TripletMatch(group, *(selected[n] for n in COMPONENTS)))
        audit.append(
            {"group_id": group, "matched": not problems, "reasons": problems, "components": details}
        )
    qc = {
        "policy": policy,
        "groups": audit,
        "matched_groups": len(matches),
        "excluded_groups": len(groups) - len(matches),
        "unassigned_sequences": {n: sorted(available[n] - assigned[n]) for n in COMPONENTS},
    }
    return matches, qc
