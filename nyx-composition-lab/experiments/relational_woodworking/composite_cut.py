"""Finite relational import experiment, not a machine controller.
HUMAN: compare local cut plans by reusable composite material.
AI: source = De Referentiis Modularibus; synthetic observable O only.
Domain: rectangular source records with integer mm, ordered layer labels.
No physical H_perp bridge, strength prediction, or full Morcillien claimed.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
import json


@dataclass(frozen=True)
class Blank:
    x: int
    y: int
    layers: tuple = (0, 1, 0)  # undirected fibre axes in the local source frame

    @property
    def area(self):
        return self.x * self.y

    def rotate(self, k):
        # C4 is a discrete chart action; no physical Galois calibration assumed.
        return Blank(self.y if k % 2 else self.x,
                     self.x if k % 2 else self.y,
                     tuple((a + k) % 2 for a in self.layers))


def split(blank, axis, length, kerf=3):
    """HUMAN: one full guillotine cut, measured from a local edge.
    AI: require 0 < length and length+kerf < extent; otherwise STOP.
    Preserve layer order, dimensions and exact material ledger in O.
    """
    if axis not in (0, 1) or type(length) is not int or kerf <= 0:
        raise ValueError('STOP: invalid cut')
    extent = (blank.x, blank.y)[axis]
    if not 0 < length < extent - kerf:
        raise ValueError('STOP: outside cut envelope')
    rest = extent - length - kerf
    if axis == 0:
        return (Blank(length, blank.y, blank.layers),
                Blank(rest, blank.y, blank.layers), kerf * blank.y)
    return (Blank(blank.x, length, blank.layers),
            Blank(blank.x, rest, blank.layers), kerf * blank.x)


def plan(kind):
    stock = Blank(1200, 600)
    # First cut separates the strip from the main reusable offcut.
    a = 1 if kind == 'A' else 0
    b = 1 - a
    strip, main, loss0 = split(stock, a, (590, 280)[a])
    first, strip, loss1 = split(strip, b, (590, 280)[b])
    second, tail, loss2 = split(strip, b, (590, 280)[b])
    return stock, (first, second), (main, tail), loss0 + loss1 + loss2


def residue(r, depth=1):
    m = 17 ** depth
    if r.denominator % 17 == 0:
        return {'status': 'STOP', 'reason': 'denominator non-invertible'}
    return r.numerator * pow(r.denominator, -1, m) % m


def fit(stock, need, k, kerf=3):
    """Check a local two-cut extraction; an equal dimension needs no cut.
    Preserve exact ratios and ordered layer directions, never just residues.
    """
    p = stock.rotate(k)
    ratios = (Q(need.x, p.x), Q(need.y, p.y))
    dx, dy = p.x - need.x, p.y - need.y
    dimensions = all(d == 0 or d >= kerf for d in (dx, dy))
    layers = p.layers == need.layers
    return {'ratios': [str(r) for r in ratios],
            'mod17': [residue(r) for r in ratios],
            'dimensions': dimensions, 'layers': layers,
            'admissible': dimensions and layers}


def run():
    needs = {'wide': Blank(700, 300), 'square': Blank(500, 500),
             'parallel_layers': Blank(500, 500, (0, 0, 0))}
    result = {'source': 'synthetic three-layer panel, not measured data',
              'plans': {}}
    for kind in ('A', 'B'):
        stock, parts, scraps, loss = plan(kind)
        result['plans'][kind] = {
            'parts': [[p.x, p.y] for p in parts],
            'scraps': [[p.x, p.y] for p in scraps],
            'kerf_area': loss,
            'scrap_area': sum(p.area for p in scraps),
            'fits': {name: [{'scrap': i, 'turn': k, **fit(p, need, k)}
                            for i, p in enumerate(scraps) for k in range(4)]
                     for name, need in needs.items()}}
    return result


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
