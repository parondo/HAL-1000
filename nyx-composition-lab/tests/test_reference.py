"""Independent finite checks; no probabilistic assumptions."""
from fractions import Fraction
import unittest

from nyx.reference import DomainStop, modular_ratio


class ReferenceTests(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(modular_ratio(Fraction(3, 5), 1), 4)
        self.assertEqual(modular_ratio(Fraction(17, 34), 1), 9)

    def test_collision_and_separation(self):
        a, b = Fraction(590, 1200), Fraction(607, 1200)
        self.assertNotEqual(a, b)
        self.assertEqual([modular_ratio(x, 1) for x in (a, b)], [8, 8])
        self.assertEqual([modular_ratio(x, 2) for x in (a, b)], [263, 178])

    def test_stop(self):
        for ratio, depth in [(Fraction(1, 17), 1), (Fraction(1), 0), (0.5, 1)]:
            with self.assertRaises(DomainStop):
                modular_ratio(ratio, depth)

    def test_multiplication(self):
        a, b, c = Fraction(1, 2), Fraction(7, 15), Fraction(7, 30)
        for n in (1, 2, 3):
            m = 17 ** n
            self.assertEqual(modular_ratio(a, n) * modular_ratio(b, n) % m,
                             modular_ratio(c, n))

    def test_finite_window(self):
        ratios = {Fraction(a, b) for a in range(-16, 17)
                  for b in range(1, 17)}
        self.assertEqual(len(ratios), 319)
        residues = {modular_ratio(r, 3) for r in ratios}
        self.assertEqual(len(residues), len(ratios))
        # Verify inverse by the congruence, not by calling pow again.
        for r in ratios:
            self.assertEqual((modular_ratio(r, 3) * r.denominator - r.numerator)
                             % (17 ** 3), 0)


if __name__ == "__main__":
    unittest.main()
