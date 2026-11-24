"""V2 phase windows derived from approved unit normals.

A normal ``n`` is a rational unit complex number.  A real unit phase ``p``
passes its check when the real inner product ``dot(p, n) >= 0``, i.e. when
``p`` lies on the closed half circle centered on ``n``.
"""
from dataclasses import dataclass
from fractions import Fraction

from calibration.quadratures import RationalComplex
from measurements.readings import parse_rational


@dataclass(frozen=True)
class ArcWindow:
    """Counterclockwise closed arc from ``start`` to ``end``, length <= pi."""

    start: RationalComplex
    end: RationalComplex

    @property
    def kind(self) -> str:
        return 'arc'

    def as_dict(self) -> dict:
        return {
            'kind': 'arc',
            'start': [str(self.start.real), str(self.start.imag)],
            'end': [str(self.end.real), str(self.end.imag)],
        }


@dataclass(frozen=True)
class PointsWindow:
    """Two antipodal rational unit points; only the points themselves count."""

    first: RationalComplex
    second: RationalComplex

    @property
    def kind(self) -> str:
        return 'points'

    def as_dict(self) -> dict:
        points = sorted((self.first, self.second), key=lambda value: (value.real, value.imag))
        return {
            'kind': 'points',
            'points': [[str(value.real), str(value.imag)] for value in points],
        }


PhaseWindow = ArcWindow | PointsWindow


def _cross(left: RationalComplex, right: RationalComplex) -> Fraction:
    return left.real * right.imag - left.imag * right.real


def _neg(value: RationalComplex) -> RationalComplex:
    return RationalComplex(-value.real, -value.imag)


def parse_normal(value: object) -> RationalComplex:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('each normal requires real and imaginary components')
    normal = RationalComplex(parse_rational(value[0]), parse_rational(value[1]))
    if normal.norm_squared() != 1:
        raise ValueError('each normal must be a unit complex number')
    return normal


def parse_normals(normals: object) -> tuple[RationalComplex, ...]:
    if isinstance(normals, (str, bytes)) or not isinstance(normals, (list, tuple)):
        raise ValueError('normals must be one or two component pairs')
    if not 1 <= len(normals) <= 2:
        raise ValueError('a turn artifact requires one or two normals')
    parsed = tuple(parse_normal(normal) for normal in normals)
    return parsed


def window_from_normals(normals: object) -> PhaseWindow:
    return window_from_parsed_normals(parse_normals(normals))


def window_from_parsed_normals(parsed: tuple[RationalComplex, ...]) -> PhaseWindow:
    negative_i = RationalComplex(0, -1)
    positive_i = RationalComplex(0, 1)
    if len(parsed) == 1 or parsed[0] == parsed[1]:
        normal = parsed[0]
        return ArcWindow(negative_i * normal, positive_i * normal)
    first, second = parsed
    cross = _cross(first, second)
    if cross == 0:
        # Distinct unit vectors with zero cross product are antipodal.
        if second != _neg(first):
            raise ValueError('opposite normals must be exact antipodes')
        p_first = negative_i * first
        p_second = positive_i * first
        ordered = sorted((p_first, p_second), key=lambda value: (value.real, value.imag))
        return PointsWindow(ordered[0], ordered[1])
    if cross < 0:
        # Keep the shorter (positive-length, at most half-circle) orientation.
        first, second = second, first
    # Half-plane of n2 starts at -i*n2; half-plane of n1 ends at +i*n1.
    return ArcWindow(negative_i * second, positive_i * first)


def _parse_unit_point(value: object) -> RationalComplex:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('phase window points require real and imaginary text')
    if not all(isinstance(part, str) for part in value):
        raise ValueError('phase window coordinates must be rational text')
    point = RationalComplex(parse_rational(value[0]), parse_rational(value[1]))
    if point.norm_squared() != 1:
        raise ValueError('phase window points must be unit complex numbers')
    return point


def decode_window(body: object) -> PhaseWindow:
    if not isinstance(body, dict):
        raise ValueError('phase window must be an object')
    kind = body.get('kind')
    if kind == 'arc':
        if body.keys() != {'kind', 'start', 'end'}:
            raise ValueError('invalid arc window fields')
        start = _parse_unit_point(body['start'])
        end = _parse_unit_point(body['end'])
        if start == end:
            raise ValueError('arc window requires distinct endpoints')
        if _cross(start, end) < 0:
            raise ValueError('arc window must be counterclockwise and at most half a turn')
        return ArcWindow(start, end)
    if kind == 'points':
        if body.keys() != {'kind', 'points'}:
            raise ValueError('invalid points window fields')
        raw_points = body['points']
        if not isinstance(raw_points, list) or len(raw_points) != 2:
            raise ValueError('points window requires exactly two points')
        points = sorted(
            (_parse_unit_point(value) for value in raw_points),
            key=lambda value: (value.real, value.imag),
        )
        if points[0] == points[1] or points[1] != _neg(points[0]):
            raise ValueError('points window requires two antipodal unit points')
        return PointsWindow(points[0], points[1])
    raise ValueError('unknown phase window kind')


def window_points(window: PhaseWindow) -> tuple[RationalComplex, ...]:
    if isinstance(window, ArcWindow):
        return (window.start, window.end)
    return (window.first, window.second)
