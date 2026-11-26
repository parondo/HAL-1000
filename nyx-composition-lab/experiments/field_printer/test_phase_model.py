"""Alternative integer-difference certificate; no numerical phase approximation.
Run: python -m unittest -v
This is an implementation cross-check, not an external audit.
"""
import json
from collections import Counter
from itertools import product
from pathlib import Path
import unittest
import HFP17_phase_simulation as model


def difference_histogram(q):
    # Independent formula: unordered pairs contribute both signed differences.
    h = Counter({0: len(q)})
    for i, a in enumerate(q):
        for b in q[i+1:]:
            h[(a-b)%17] += 1
            h[(b-a)%17] += 1
    return tuple(h[i] for i in range(17))


class Audit(unittest.TestCase):
    def test_all_relative_phase_pairs(self):
        for a,b in product(range(17), repeat=2):
            q=(0,a,b)
            h=difference_histogram(q)
            self.assertEqual(sum(h),9)
            self.assertEqual(model.intensity(q),tuple(x-h[16] for x in h[:16]))
            self.assertEqual(h[0]==9,a==b==0)

    def test_trace_fibres(self):
        counts=Counter(difference_histogram((0,a,b))
                       for a,b in product(range(17),repeat=2))
        self.assertEqual(len(counts),33)
        self.assertEqual(Counter(counts.values()),{1:1,6:16,12:16})
        self.assertEqual(sum(counts.values()),289)

    def test_reproduction(self):
        expected=json.loads(Path(__file__).with_name(
            'HFP17_phase_results.json').read_text())
        self.assertEqual(json.loads(json.dumps(model.run())),expected)

    def test_galois_is_not_value_invariance(self):
        # The maximum predicate is invariant, not every trace value.
        self.assertNotEqual(model.intensity((0,1,4)),
                            model.intensity((0,2,8)))


if __name__ == '__main__':
    unittest.main()
