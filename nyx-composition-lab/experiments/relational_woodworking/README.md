# Relational woodworking: composite-panel witness

Human entry point: two local cut sequences deliver the same two 590 × 280 mm parts from a 1200 × 600 mm panel, with 3 mm kerf. The synthetic composite has three ordered layers with undirected orientation labels (0,1,0). The external request describes dimensions and required layer labels, not a shared global origin.

Run from this directory (Python 3.9+, standard library only):

```sh
python -m unittest -v
python composite_cut.py
```

| Plan | Main offcut | Secondary offcut | Kerf area mm² | Reusable area mm² |
|---|---|---|---:|---:|
| A | 1200 × 317 | 14 × 280 | 5280 | 384320 |
| B | 607 × 600 | 590 × 34 | 5340 | 384260 |

A accepts the 700 × 300 request; B accepts the 500 × 500 request. Neither accepts the all-parallel layer requirement. The square is not orientation-independent when ordered layer directions are specified. The two allowed chart orientations are 0 and 2, which coincide in this simplified undirected model.

AI contract:
- Source: De Referentiis Modularibus, synthetic observable source records. Read ../../AGENTS.md and ../../docs/reference-import.md.
- Exact rational fit ratios are retained alongside modular signatures. A residue is not a dimension, capacity, or admissibility predicate.
- The four chart turns compose by addition mod 4; layer orientations are undirected labels mod 2. This is a source chart action, not a calibrated physical Galois action.
- Preconditions: positive integer dimensions and kerf, declared rectangular source frame, ordered binary layer directions, two prescribed full-cut sequences. Other stock geometries and operations are not modeled.
- A fit means only that the specified extraction is dimensionally and layer-label compatible under the simplified kerf rule. It is not a physical safety or strength certificate.
- Local tests: five test methods passed. All 48 (plan, offcut, request, turn) evaluations are enumerated deterministically; no sampling or probability.
- STOP for noninvertible modular denominators; preserve the exact ratio. No bridge to H_perp, no calibrated full Morcillien, no claim of physical optimization, no inter-tool language. No composite constitutive law is imported.
- A future port should reuse the common rational residue operator; the tiny standalone evaluator is duplicated here solely for reproducibility of this witness.
