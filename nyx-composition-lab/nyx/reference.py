"""Exact rational reference import. No physical geometry is inferred."""
from fractions import Fraction


class DomainStop(ValueError):
    """The operation is outside its declared domain; not a false value."""


def modular_ratio(ratio: Fraction, depth: int) -> int:
    """HUMAN: compute a modular signature, retaining the exact input separately.

    AI CONTRACT: docs/operator-contract.json.
    Domain: reduced Fraction and positive integer depth.
    Invariant: multiplicative composition.
    Non-invariant: absolute magnitude or physical form.
    STOP: invalid depth, type, or non-invertible reduced denominator.
    """
    if not isinstance(ratio, Fraction):
        raise DomainStop("STOP: expected an exact Fraction")
    if type(depth) is not int or depth < 1:
        raise DomainStop("STOP: depth must be a positive integer")
    if ratio.denominator % 17 == 0:
        raise DomainStop("STOP: reduced denominator not invertible")
    modulus = 17 ** depth
    return ratio.numerator * pow(ratio.denominator, -1, modulus) % modulus
