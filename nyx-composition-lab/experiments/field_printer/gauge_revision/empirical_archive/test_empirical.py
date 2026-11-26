"""Independent numerical enclosure check (external optical adapter only).
No fitted coefficients, Gaussian metrics, or interpolation of measurements.
The Decimal recurrence is an auxiliary implementation audit, not a Nyx primitive.
"""
from decimal import Decimal as D, localcontext
from fractions import Fraction as Q
from itertools import product
import json
from pathlib import Path
import unittest
import empirical_check as model


def as_decimal(q):
    q=Q(q)
    return D(q.numerator)/D(q.denominator)


def trig(deg):
    # Independent series recurrence; no reuse of model's Machin implementation.
    pi=D('3.1415926535897932384626433832795028841971693993751')
    x=pi*D(deg)/180
    term_s=x; term_c=D(1); s=term_s; c=term_c
    for k in range(1,28):
        term_s*= -x*x/D((2*k)*(2*k+1))
        term_c*= -x*x/D((2*k-1)*(2*k))
        s+=term_s; c+=term_c
    return s,c


class EmpiricalAudit(unittest.TestCase):
    def test_reference_json(self):
        expected=json.loads(Path(__file__).with_name('results.json').read_text())
        actual=json.loads(json.dumps(model.run()))
        expected.pop('python'); actual.pop('python')
        self.assertEqual(expected,actual)

    def test_prediction_enclosures(self):
        with localcontext() as ctx:
            ctx.prec=48
            for row in model.run()['rows']:
                s,c=trig(row['angle_deg']); lam=D(row['wavelength'])
                k=row['kind']
                if k=='tilt': value=D(row['travel'])*s/c
                elif k=='axial': value=lam/(1-c)
                elif k=='lateral': value=lam/s
                elif k=='radial5': value=lam/(2*s*trig(54)[1])
                lo,hi=map(as_decimal,row['predicted_interval'])
                self.assertLessEqual(lo,value); self.assertLessEqual(value,hi)
                error=(D(row['observed'])/value-1)*100
                a,b=map(as_decimal,row['signed_relative_percent_interval'])
                self.assertLessEqual(a,error); self.assertLessEqual(error,b)

    def test_full_atlas_by_constructing_fibres(self):
        # Construct preimages, instead of testing the predicate in a loop.
        good=(0,1,2,3,4,13,14,15,16)
        fibres={k:{(x,y,(x+y-k)%17) for x,y in product(range(17),repeat=2)}
                for k in range(17)}
        union=set().union(*(fibres[k] for k in good))
        self.assertEqual(len(union),2601)
        self.assertEqual(sum(len(s) for s in fibres.values()),4913)
        self.assertEqual(len(set().union(*fibres.values())),4913)
        self.assertEqual(len(union & set(product((0,8),repeat=3))),4)

    def test_pair_phase_budget(self):
        # Derive pair weight sum independently from the four polarizations.
        e=((0,101,0),(0,101,0),(0,101,0),(0,99,-20))
        weights=sum(Q(sum(a*b for a,b in zip(e[i],e[j])),101**2)
                    for i in range(4) for j in range(i+1,4))
        eps=Q(model.run()['phase_budget']['absolute_pair_error_turns_strict_upper'])
        self.assertEqual(weights,Q(600,101))
        self.assertEqual(4*Q(22,7)*weights*eps,Q(297,1616))
        self.assertEqual(eps*405,Q(5103,5120))


if __name__=='__main__':
    unittest.main()
