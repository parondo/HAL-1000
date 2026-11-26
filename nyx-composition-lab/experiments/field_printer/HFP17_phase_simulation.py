"""HFP-17: exact synthetic phase-register experiment, NOT a physical simulator.
Human: two relative phases select a target address in a finite 17 x 17 atlas.
AI: all arithmetic is finite cyclotomic arithmetic in Q[z]/Phi_17.
No physical position, dose, chirality or material law is inferred.
"""
from itertools import product
import json
from pathlib import Path

P = 17
ADDR = list(product(range(P), repeat=2))
TARGET = (4, 13)  # declared synthetic reference, not measured

# Canonical coefficient representation: eliminate z^16 via Phi_17=0.
def intensity(phases):
    coefficients = [0] * P
    for a in phases:
        for b in phases:
            coefficients[(a-b) % P] += 1
    return tuple(c - coefficients[-1] for c in coefficients[:-1])


def phases(point, target=TARGET):
    x, y = point
    a, b = target
    return (0, (x-a) % P, (y-b) % P)


def exact_max(n):
    return (n*n,) + (0,) * 15


def run():
    fields = {p: phases(p) for p in ADDR}
    one = [p for p, q in fields.items() if intensity(q[:2]) == exact_max(2)]
    two = [p for p, q in fields.items() if intensity(q[::2]) == exact_max(2)]
    three = [p for p, q in fields.items() if intensity(q) == exact_max(3)]
    assert len(one) == len(two) == 17 and three == [TARGET]
    checks = dict(global_phase=0, conjugation=0, galois_max=0, translation=0)
    for p, q in fields.items():
        I = intensity(q)
        for h in range(P):
            assert intensity(tuple(a+h for a in q)) == I
            checks['global_phase'] += 1
        assert intensity(tuple(-a for a in q)) == I
        checks['conjugation'] += 1
        for k in range(1, P):
            assert (intensity(tuple(k*a for a in q)) == exact_max(3)) == (p == TARGET)
            checks['galois_max'] += 1
    for dx, dy in ADDR:
        target = ((TARGET[0]+dx)%P, (TARGET[1]+dy)%P)
        shifted = [p for p in ADDR if intensity(phases(p,target)) == exact_max(3)]
        assert shifted == [target]
        checks['translation'] += 1
    # Distinct signed phase patterns with identical quadratic trace.
    a, b = (0, 1, 4), (0, 16, 13)
    assert a != b and intensity(a) == intensity(b)
    return {'scope':'synthetic finite algebra only', 'target':TARGET,
            'atlas_addresses':len(ADDR), 'two_beam_x':len(one),
            'two_beam_y':len(two), 'three_beam':three,
            'checks':checks,'total_invariance_checks':sum(checks.values()),
            'conjugate_witness':{'phases_a':a,'phases_b':b,'trace':intensity(a)}}

if __name__ == '__main__':
    result=run()
    print(json.dumps(result, indent=2))
    Path(__file__).with_name('HFP17_phase_results.json').write_text(json.dumps(result, indent=2)+'\n')
