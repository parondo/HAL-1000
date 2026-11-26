"""Independent exact checks on the declared source model."""
import unittest
from fractions import Fraction as Q
from composite_cut import Blank, plan, run, fit, residue


class TestComposite(unittest.TestCase):
    def test_ledger(self):
        for k, expected_loss in [('A', 5280), ('B', 5340)]:
            stock, parts, scraps, loss = plan(k)
            self.assertEqual(loss, expected_loss)
            self.assertEqual(parts, (Blank(590, 280),) * 2)
            self.assertEqual(sum(p.area for p in parts + scraps) + loss, stock.area)

    def test_expected_scraps(self):
        self.assertEqual(plan('A')[2], (Blank(1200, 317), Blank(14, 280)))
        self.assertEqual(plan('B')[2], (Blank(607, 600), Blank(590, 34)))

    def test_turn_composition(self):
        for p in (Blank(1200, 600), Blank(607, 600)):
            for i in range(4):
                for j in range(4):
                    self.assertEqual(p.rotate(i).rotate(j), p.rotate((i+j)%4))

    def test_lateral_constraints(self):
        r = run()['plans']
        for k, expected in [('A', {'wide'}), ('B', {'square'})]:
            self.assertEqual({name for name, fs in r[k]['fits'].items()
                              if any(f['admissible'] for f in fs)}, expected)
        self.assertTrue(fit(Blank(607, 600), Blank(500, 500), 0)['admissible'])
        self.assertFalse(fit(Blank(607, 600), Blank(500, 500), 1)['admissible'])

    def test_unit_invariance_and_stop(self):
        r = Q(700, 1200)
        self.assertEqual(r, Q(70, 120))
        self.assertEqual(r, (Q(700)*Q(5,127))/(Q(1200)*Q(5,127)))
        self.assertEqual(residue(r), residue(Q(70,120)))
        self.assertEqual(residue(Q(1,17))['status'], 'STOP')


if __name__ == '__main__':
    unittest.main()
