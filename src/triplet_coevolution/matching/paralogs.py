"""Resolve duplicate component candidates only with explicit evidence priorities."""


def select_candidate(candidates, policy):
    if len(candidates) == 1:
        return candidates[0].sequence_id
    if not candidates or policy != "priority" or any(c.priority is None for c in candidates):
        return None
    best = max(c.priority for c in candidates)
    winners = [c.sequence_id for c in candidates if c.priority == best]
    return winners[0] if len(winners) == 1 else None
