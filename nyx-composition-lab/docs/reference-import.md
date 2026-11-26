# Exact reference import

Sources: Scala Caelestis V4, section 3.11 (Scriptura Pulveris), and the WordPress notebook De Referentiis Modularibus. Corpus index: https://www.fdaa.fr/. Exact public section links remain unpinned.

A measurement, its representation, and its physical form are distinct. A rational ratio can be transported without importing a physical ontology.

For a reduced rational a/b with 17 not dividing b, define E_n(a/b) = a * b^(-1) mod 17^n, for positive integer n. The reduced denominator must be invertible. Otherwise the evaluator returns a typed STOP. Multiplication is preserved because the denominators are invertible and multiplication of residues respects the same rational identity.

Example: 590/1200 and 607/1200 both evaluate to 8 mod 17, but are different ratios. At modulus 289 they evaluate to 263 and 178. Keep the exact ratio and reference; do not substitute the signature for either.

## Finite reconstruction bound

Take two reduced fractions a/b and c/d, |a|,|c| <= A and 1 <= b,d <= B. Assume both denominators are invertible modulo 17. If they share a residue modulo m=17^n, then m divides ad-bc. Since |ad-bc| <= 2AB, m > 2AB forces ad-bc=0. Thus the residue is injective on the declared finite window. This is a sufficient bound, not a universal minimal depth.

The test suite checks a small finite window as a witness, not a replacement for the proof.

## Scope

No equivalence with the cyclotomic field or a physical H_perp realization is asserted. No capacity, Morcillien calibration, machine sensor model or inter-tool language is fabricated. This is the rational import layer. The source notebook retains the synthetic woodworking example; a full port of its checks is separate work.
