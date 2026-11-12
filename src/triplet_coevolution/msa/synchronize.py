
# synchronize 3 MSAs using an explicit homolog matching table

from collections.abc import Iterable

from triplet_coevolution.core.alignment import (
  ComponentMSA,
  MatchedTripletMSA,
  TripletMatch,
)
from triplet_coevolution.io.metadata import TripletTable


# accept either the full metadata table or a sequence of match records
def _get_matches(
  matches: TripletTable | Iterable[TripletMatch],
) -> tuple[TripletMatch, ...]:
  """return validated one to one matching records"""

  if isinstance(matches, TripletTable):
    return matches.matches

  # TripletTable validates missing IDs and duplicate assignments
  return TripletTable(matches=tuple(matches)).matches


# check component labels and ids before attempting reordering
def _check_inputs(
  a: ComponentMSA,
  b: ComponentMSA,
  c: ComponentMSA,
  matches: tuple[TripletMatch, ...],
) -> None:
  """reject wrong component order and missing referenced homologs"""

  if (a.name, b.name, c.name) != ("A", "B", "C"):
    raise ValueError("components must be named A, B, C in this order")

  for component, field in (
    (a, "a_id"),
    (b, "b_id"),
    (c, "c_id"),
  ):
    available = set(component.ids)
    missing = [
      getattr(match, field)
      for match in matches
      if getattr(match, field) not in available
    ]

    if missing:
      raise ValueError(
        f"component {component.name} is missing matched IDs: {missing}"
      )


# identify extra hits without modifying the original MSAs
def _unmatched_ids(
  a: ComponentMSA,
  b: ComponentMSA,
  c: ComponentMSA,
  records: tuple[TripletMatch, ...],
) -> dict[str, tuple[str, ...]]:
  result = {}

  for component, field in (
    (a, "a_id"),
    (b, "b_id"),
    (c, "c_id"),
  ):
    selected = {getattr(match, field) for match in records}
    result[component.name] = tuple(
      seq_id for seq_id in component.ids
      if seq_id not in selected
    )

  return result


def unmatched_sequence_ids(
  a: ComponentMSA,
  b: ComponentMSA,
  c: ComponentMSA,
  matches: TripletTable | Iterable[TripletMatch],
) -> dict[str, tuple[str, ...]]:
  """find input sequences not selected for complete triplet analysis

  reports unused hits in the original MSA row order doesnt mean their partners are biologically absent
  """

  records = _get_matches(matches)
  _check_inputs(a, b, c, records)
  return _unmatched_ids(a, b, c, records)


# build a complete synchronized triplet MSA
def synchronize(
  a: ComponentMSA,
  b: ComponentMSA,
  c: ComponentMSA,
  matches: TripletTable | Iterable[TripletMatch],
  *,
  require_all: bool = False,
) -> MatchedTripletMSA:
  """select and reorder 3 alignments using explicit homolog matches

  source MSAs may have different numbers of sequences
  only complete matched observations are included in the result
  other hits remain in the original ComponentMSA objects
  require_all=True rejects any unused input hits
  """

  records = _get_matches(matches)
  _check_inputs(a, b, c, records)

  unused = _unmatched_ids(a, b, c, records)

  if require_all and any(unused.values()):
    raise ValueError(f"unmatched input sequences: {unused}")

  a_ordered = a.take_ids([match.a_id for match in records])
  b_ordered = b.take_ids([match.b_id for match in records])
  c_ordered = c.take_ids([match.c_id for match in records])

  return MatchedTripletMSA(
    a=a_ordered,
    b=b_ordered,
    c=c_ordered,
    matches=records,
  )
