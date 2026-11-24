import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from measurements.bands import require_name
from measurements.readings import parse_rational
from phase_evidence.turn_response import TurnEvidence
from receiving.lookup import lookup_received

from .turn_directions import solve_directions


@dataclass(frozen=True)
class TurnReviewRequest:
    receipt_ids: tuple[str, str]
    weights: tuple[tuple[str, Fraction], ...]
    energy_limit: Fraction


def _validate_weights(weights: Mapping[str, object]) -> tuple[tuple[str, Fraction], ...]:
    if not isinstance(weights, Mapping):
        raise ValueError('weights must map bands to rational values')
    parsed = []
    for key, value in weights.items():
        band = require_name(key, 'weight band')
        weight = parse_rational(value)
        if weight < 0:
            raise ValueError(f'negative weight for band {band!r}')
        parsed.append((band, weight))
    if not parsed:
        raise ValueError('at least one weighted band is required')
    if not any(weight > 0 for _, weight in parsed):
        raise ValueError('at least one weight must be positive')
    return tuple(parsed)


def validate_turn_request(
    conn: sqlite3.Connection,
    receipt_ids: tuple[str, str] | list[str],
    weights: Mapping[str, object],
    energy_limit: object,
) -> tuple[TurnReviewRequest, TurnEvidence, object]:
    if not isinstance(receipt_ids, (list, tuple)) or len(receipt_ids) != 2:
        raise ValueError('turn review requires exactly two ordered receipt ids')
    left_id, right_id = (require_name(value, 'receipt id') for value in receipt_ids)
    parsed_weights = _validate_weights(weights)
    limit = parse_rational(energy_limit)
    if limit < 0:
        raise ValueError('energy limit must be non-negative')

    left = lookup_received(conn, left_id)
    right = lookup_received(conn, right_id)
    if left is None:
        raise ValueError(f'unknown receipt {left_id!r}')
    if right is None:
        raise ValueError(f'unknown receipt {right_id!r}')
    if left.source.station == right.source.station:
        raise ValueError('turn review requires artifacts from different stations')
    if left.delivery.version != 2:
        raise ValueError('the left artifact must be a V2 turn artifact')
    if right.delivery.version != 1:
        raise ValueError('the right artifact must be an anchored V1 artifact')
    right_evidence = right.evidence
    if not right_evidence.anchored:
        raise ValueError('the right artifact must be anchored to the common reference')

    left_evidence = left.evidence
    left_bands = {band for band, _ in left_evidence.responses}
    right_bands = {band for band, _ in right_evidence.responses}
    for band, _ in parsed_weights:
        if band not in left_bands or band not in right_bands:
            raise ValueError(f'request band {band!r} is missing from at least one artifact')
    request = TurnReviewRequest((left_id, right_id), parsed_weights, limit)
    return request, left, right


def build_turn_report(request: TurnReviewRequest, left_item, right_item,
                      left_evidence: TurnEvidence, right_evidence) -> dict:
    directions = solve_directions(
        left_evidence, right_evidence.responses, request.weights, request.energy_limit
    )
    return {
        'request': {
            'receipt_ids': list(request.receipt_ids),
            'weights': {band: str(weight) for band, weight in request.weights},
            'energy_limit': str(request.energy_limit),
        },
        'parents': [
            {'artifact_id': left_item.receipt.artifact_id, 'source': left_item.source.as_dict()},
            {'artifact_id': right_item.receipt.artifact_id, 'source': right_item.source.as_dict()},
        ],
        'directions': directions,
    }
