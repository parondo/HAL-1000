"""Synthetic local-reference example, with exact rational arithmetic."""
from fractions import Fraction
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nyx.reference import modular_ratio


def run() -> dict:
    # INPUT PROVENANCE: De Referentiis Modularibus, synthetic millimetre lengths.
    first, second = Fraction(590, 1200), Fraction(607, 1200)
    ab, bc, ac = Fraction(1, 2), Fraction(7, 15), Fraction(7, 30)
    # Verify exact ratio composition before inspecting modular signatures.
    if ab * bc != ac:
        raise RuntimeError("Exact composition failed")
    return {
        "input_kind": "synthetic",
        "ratios": [str(first), str(second)],
        "mod17": [modular_ratio(first, 1), modular_ratio(second, 1)],
        "mod289": [modular_ratio(first, 2), modular_ratio(second, 2)],
        "composition": [modular_ratio(x, 1) for x in (ab, bc, ac)],
        "scope": "rational import only; no physical bridge established",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
