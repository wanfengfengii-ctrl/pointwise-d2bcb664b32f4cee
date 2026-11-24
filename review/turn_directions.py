"""Exact safe-direction solver for V2 turn reviews.

For a left (unanchored, V2) response ``a_k`` and a right anchored response
``b_k`` in band ``k``, the knob unit complex number ``t`` and the actual
phase ``p`` (constrained by the artifact's phase window) produce energy

    E(t, p) = sum_k w_k |t p a_k + b_k|^2
            = B0 + 2 Re(t p C),

with rational constants

    B0 = sum_k w_k (|a_k|^2 + |b_k|^2),
    C  = sum_k w_k a_k conj(b_k),   D = |C|^2.

Writing ``H = (L - B0) / 2`` for the non-negative rational limit ``L``, every
constraint has the half-plane form ``Re(t g) <= H`` with ``g = p C``.  On the
unit circle the boundary of ``Re(t g) = H`` is the pair

    t_pm = (H +/- i*sqrt(M)) / sqrt(D) * conj(g) / sqrt(D),   M = D - H^2,

and ``H^2 + M = D`` together with ``|conj(g)| = sqrt(D)`` keeps both points
on the unit circle.

Arc windows forbid one contiguous neighborhood of knob directions, so the
safe set is at most one closed arc; antipodal-point windows produce a strip
intersection that yields two points or two maximal closed arcs.  Every
endpoint stays an exact quadratic radical; no floats are used.
"""
from dataclasses import dataclass
from fractions import Fraction

from calibration.quadratures import RationalComplex
from phase_evidence.turn_response import TurnEvidence
from phase_evidence.window import ArcWindow


@dataclass(frozen=True)
class Radial:
    """A coordinate value ``a + b * sqrt(d)`` with rational a, b and integer d."""

    a: Fraction
    b: Fraction
    d: int

    def as_dict(self) -> dict[str, str]:
        return {'a': str(self.a), 'b': str(self.b), 'd': str(self.d)}


def _squarefree_integer(value: int) -> tuple[int, int]:
    """Return ``(d, s)`` for positive integer ``value`` so value == d * s**2, d squarefree."""
    reduced = value
    square_factor = 1
    prime = 2
    while prime * prime <= reduced:
        while reduced % (prime * prime) == 0:
            reduced //= prime * prime
            square_factor *= prime
        prime += 1 if prime == 2 else 2
    return reduced, square_factor


def squarefree_parts(value: Fraction) -> tuple[int, Fraction]:
    """Return ``(d, f)`` so that ``sqrt(value) == f * sqrt(d)``, d squarefree.

    A rational result is reported as ``(1, rational root)``; zero as
    ``(0, 0)``.
    """
    if value < 0:
        raise ValueError('radicand must be non-negative')
    if value == 0:
        return 0, Fraction(0)
    numerator, denominator = value.numerator, value.denominator
    num_d, num_s = _squarefree_integer(numerator)
    den_d, den_s = _squarefree_integer(denominator)
    if num_d == den_d:
        return 1, Fraction(num_s, den_s)
    # sqrt(n/m) = (num_s/den_s) * sqrt(num_d/den_d)
    #           = (num_s/(den_s*den_d)) * sqrt(num_d * den_d); the product of
    # two distinct squarefree integers is again squarefree.
    return num_d * den_d, Fraction(num_s, den_s * den_d)


def _coordinate(rational_part: Fraction, radical_factor: Fraction,
                radical_radicand: Fraction) -> dict:
    """Encode ``rational_part + radical_factor * sqrt(radical_radicand)``.

    ``radical_factor`` carries the sign and the radicand stays non-negative.
    When the radicand's square root is rational it folds into the rational
    part; otherwise the single squarefree radical is reported explicitly.
    """
    if radical_radicand == 0 or radical_factor == 0:
        return Radial(rational_part, Fraction(0), 0).as_dict()
    d_value, factor = squarefree_parts(radical_radicand)
    if d_value == 1:
        return Radial(rational_part + radical_factor * factor, Fraction(0), 0).as_dict()
    return Radial(rational_part, radical_factor * factor, d_value).as_dict()


def _point_dict(sign_im: int, h_param: Fraction, m: Fraction, d_param: Fraction,
                conjugate_g: RationalComplex) -> dict:
    # The unit-circle boundary of Re(t g) = H is, with M = D - H^2 and
    # s = +/- 1,
    #   t_s = (H + s*i*sqrt(M)) / sqrt(D) * conj(g) / sqrt(D).
    # Writing gx = Re(conj(g)), gy = Im(conj(g)) this expands to
    #   real = H*gx/D - s*sqrt(M)*gy/D
    #   imag = H*gy/D + s*sqrt(M)*gx/D.
    # Each coordinate is one rational part plus at most one quadratic radical,
    # and |t_s| = 1 because H^2 + M = D and |conj(g)| = sqrt(D).
    gx = conjugate_g.real
    gy = conjugate_g.imag
    radical_factor_scale = 1 / d_param
    return {
        'real': _coordinate(
            h_param * gx / d_param,
            -sign_im * gy * radical_factor_scale,
            m,
        ),
        'imag': _coordinate(
            h_param * gy / d_param,
            sign_im * gx * radical_factor_scale,
            m,
        ),
    }


def _direction(point: dict) -> list[dict]:
    return [point['real'], point['imag']]


def _single_point(g_mid: RationalComplex) -> list[dict]:
    """Encode the unit direction ``-conj(g_mid) / |g_mid|`` exactly.

    At the feasibility boundary the safe set collapses to the direction
    opposite the midpoint of the forbidden g-arc; dividing by its own modulus
    keeps the result on the unit circle.
    """
    conjugate_mid = g_mid.conjugate()
    mid_norm_squared = g_mid.norm_squared()

    def reciprocal_sqrt(factor: Fraction, radicand: Fraction) -> dict:
        # factor / sqrt(radicand): if sqrt(radicand) = f*sqrt(d) with d > 1,
        # this is factor/(f*d) * sqrt(d); for the rational core d = 1 it is
        # simply factor/f.
        d_value, square_factor = squarefree_parts(radicand)
        if d_value == 1:
            return Radial(factor / square_factor, Fraction(0), 0).as_dict()
        return Radial(Fraction(0), factor / (square_factor * d_value), d_value).as_dict()

    return [
        reciprocal_sqrt(-conjugate_mid.real, mid_norm_squared),
        reciprocal_sqrt(-conjugate_mid.imag, mid_norm_squared),
    ]


def _arc_object(start: dict, end: dict) -> dict:
    return {'start': _direction(start), 'end': _direction(end)}


def solve_directions(
    left: TurnEvidence,
    right_responses: tuple[tuple[str, RationalComplex], ...],
    weights: tuple[tuple[str, Fraction], ...],
    limit: Fraction,
) -> dict:
    left_values = dict(left.responses)
    right_values = dict(right_responses)
    base = Fraction(0)
    cross = RationalComplex()
    for band, weight in weights:
        a = left_values[band]
        b = right_values[band]
        base += weight * (a.norm_squared() + b.norm_squared())
        cross += RationalComplex(weight) * a * b.conjugate()
    d_param = cross.norm_squared()
    if d_param == 0:
        return {'kind': 'full' if base <= limit else 'empty'}

    # Global extrema of Re(t p C) over any unit t, p are +/- sqrt(D).  The
    # whole circle is safe at the upper bound and unsafe at the lower one.
    upper_gap = limit - base
    if upper_gap >= 0 and upper_gap * upper_gap >= 4 * d_param:
        return {'kind': 'full'}
    lower_gap = base - limit
    if lower_gap >= 0 and lower_gap * lower_gap >= 4 * d_param:
        return {'kind': 'empty'}

    h_param = upper_gap / 2
    m = d_param - h_param * h_param
    window = left.window

    if isinstance(window, ArcWindow):
        # The g(p) = p*C image is a CCW arc of half-width delta.  The smallest
        # achievable value of its maximum Re(t g) is -sqrt(D) cos(delta), so a
        # knob direction exists exactly when H >= -sqrt(D) cos(delta).  With
        # cos(2*delta) = Re(conj(g_a) g_b) / D the squared threshold is
        # D*cos^2(delta) = (D + Re(conj(g_a) g_b)) / 2.
        g_start = window.start * cross
        g_end = window.end * cross
        threshold_squared = (d_param + (g_start.conjugate() * g_end).real) / 2
        if h_param < 0 and h_param * h_param >= threshold_squared:
            if h_param * h_param > threshold_squared:
                return {'kind': 'empty'}
            # Equality: the safe set collapses to the single direction
            # opposite the midpoint of the forbidden arc.
            g_mid = g_start + g_end
            midpoint = _single_point(g_mid)
            return {'kind': 'parts', 'arcs': [], 'points': [midpoint]}
        conjugate_start = g_start.conjugate()
        conjugate_end = g_end.conjugate()
        # The safe complement of the forbidden CCW neighborhood starts on the
        # +i*sqrt(M) boundary of g_start and ends on the -i*sqrt(M) boundary of
        # g_end; both endpoints satisfy Re(t g) = H for the corresponding p.
        start = _point_dict(1, h_param, m, d_param, conjugate_start)
        end = _point_dict(-1, h_param, m, d_param, conjugate_end)
        if start == end:
            # The open safe arc has measure zero; only the shared boundary
            # direction (the +i*sqrt(M) point of g_start, which equals the
            # -i*sqrt(M) point of g_end) survives.
            return {'kind': 'parts', 'arcs': [], 'points': [_direction(start)]}
        return {'kind': 'parts', 'arcs': [_arc_object(start, end)], 'points': []}

    # Points window: two antipodal rational unit points, forming a strip.
    g1 = window.first * cross
    conjugate_1 = g1.conjugate()
    point_plus = _point_dict(1, h_param, m, d_param, conjugate_1)
    point_minus = _point_dict(-1, h_param, m, d_param, conjugate_1)
    if h_param < 0:
        # The strip |Re(t conj(g1))| <= H has no unit-circle directions.
        return {'kind': 'empty'}
    if h_param == 0:
        return {'kind': 'parts', 'arcs': [],
                'points': [_direction(point_plus), _direction(point_minus)]}
    g2 = window.second * cross
    conjugate_2 = g2.conjugate()
    point_2_plus = _point_dict(1, h_param, m, d_param, conjugate_2)
    point_2_minus = _point_dict(-1, h_param, m, d_param, conjugate_2)
    # Each maximal arc joins the upper (y > 0) and lower (y < 0) strip edges.
    # The upper safe arc runs CCW from g1's plus boundary to g2's minus
    # boundary and the lower arc from g2's plus boundary to g1's minus
    # boundary; the complementary gaps are the forbidden strip edges.
    return {
        'kind': 'parts',
        'arcs': [_arc_object(point_plus, point_2_minus),
                 _arc_object(point_2_plus, point_minus)],
        'points': [],
    }
