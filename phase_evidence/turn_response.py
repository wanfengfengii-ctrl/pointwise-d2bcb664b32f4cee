from dataclasses import dataclass

from calibration.quadratures import RationalComplex
from calibration.snapshots import FixedAnalysis

from .window import PhaseWindow, decode_window, window_from_parsed_normals


@dataclass(frozen=True)
class TurnEvidence:
    responses: tuple[tuple[str, RationalComplex], ...]
    window: PhaseWindow


def build_turn_body(analysis: FixedAnalysis, normals: tuple) -> dict:
    window = window_from_parsed_normals(normals)
    bands = [[name, str(value.real), str(value.imag)]
             for name, value in sorted(analysis.inspection.responses, key=lambda row: row[0])]
    return {'bands': bands, 'phase_window': window.as_dict()}


def decode_turn_body(body: object) -> TurnEvidence:
    if not isinstance(body, dict) or body.keys() != {'bands', 'phase_window'}:
        raise ValueError('invalid turn evidence fields')
    bands = body['bands']
    if not isinstance(bands, list) or not bands:
        raise ValueError('turn evidence bands must be a nonempty list')
    responses = []
    seen = set()
    for row in bands:
        if not isinstance(row, list) or len(row) != 3 or not all(isinstance(part, str) for part in row):
            raise ValueError('turn evidence requires band, real text, and imaginary text')
        name, real, imag = row
        if name in seen:
            raise ValueError(f'duplicate band {name!r}')
        seen.add(name)
        responses.append((name, RationalComplex(real, imag)))
    if [name for name, _ in responses] != sorted(seen):
        raise ValueError('turn evidence bands must be sorted by name')
    window = decode_window(body['phase_window'])
    return TurnEvidence(tuple(responses), window)
