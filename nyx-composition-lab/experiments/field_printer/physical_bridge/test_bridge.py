"""Independent checks of the instrument adapter, not lab measurements."""
from fractions import Fraction as Q
from itertools import product
import json
from pathlib import Path
import unittest
import bridge


class BridgeAudit(unittest.TestCase):
    def test_polynomial(self):
        # Derive the polynomial from z^k+z^-k by recurrence, not the table.
        prev, cur = [2], [0,1]
        total = [1]+[0]*8
        for k in range(1,9):
            for i,c in enumerate(cur):
                total[i] += c
            nxt = [0]+cur
            for i,c in enumerate(prev):
                nxt[i] -= c
            prev,cur = cur,nxt
        self.assertEqual(total,[1,-4,-10,10,15,-6,-7,1,1])
        self.assertTrue(all(c>0 for c in bridge.bound_certificate()))

    def test_independent_map(self):
        # Forward calculation from direction vectors, not inverse formula.
        for j in product(range(17),repeat=3):
            r=bridge.physical_point(j)
            v=tuple(sum(Q(a-b,101)*x for a,b,x in zip(d,bridge.DIRECTIONS[0],r))
                    for d in bridge.DIRECTIONS[1:])
            self.assertEqual(v,tuple(Q(a,17) for a in j))

    def test_vector_gap(self):
        self.assertEqual(bridge.MAXIMUM,Q(1604,101))
        self.assertEqual(bridge.GATE,Q(25367,1616))
        self.assertEqual(bridge.MAXIMUM-bridge.GATE,Q(297,1616))
        for q in product(range(17),repeat=3):
            phases=(0,*q)
            loss=sum(bridge.WEIGHTS[i][j]/8 for i in range(4)
                     for j in range(i+1,4) if phases[i]!=phases[j])
            if any(q):
                self.assertGreaterEqual(loss,bridge.GAP)
            else:
                self.assertEqual(loss,0)

    def test_recapture(self):
        for a in range(17):
            expected=tuple(Q(int(i==a)-int(a==16)) for i in range(16))
            self.assertEqual(bridge.recapture(a),expected)

    def test_reference_json(self):
        expected=json.loads(Path(__file__).with_name('results.json').read_text())
        actual=bridge.run()
        # Runtime version is recorded, not a mathematical invariant.
        expected.pop('python'); actual.pop('python')
        self.assertEqual(actual,expected)


if __name__=='__main__':
    unittest.main()
