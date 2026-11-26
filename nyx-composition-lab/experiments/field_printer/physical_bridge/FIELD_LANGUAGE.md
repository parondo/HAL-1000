# FieldScript: language contract, draft 0.2

The existing P0 example is declarative notation with a fixed interpreter implemented directly in Python, not a general executable DSL. The proposed physical dialect adds explicit calibration and model bindings. Do not claim that this grammar is implemented.

    program = "MODEL" name, reference, calibration, backend, target, verify, execution ;
    reference = "REFERENCE" name, "REGISTER" "Q17", "DIMENSION" "3" ;
    calibration = "WAVELENGTH_NM" ratio, "REFRACTIVE_INDEX" ratio ;
    backend = "BACKEND" "FOUR_COHERENT_VECTOR_WAVES" ;
    target = "TARGET" integer "," integer "," integer ;
    verify = "VERIFY" "PHASE_MAP" "VECTOR_GAP" "RECAPTURE" ;
    execution = "EXECUTION" "SIMULATION_ONLY" ;
    ratio = integer [ "/" positive_integer ] ;

Whitespace separates tokens. Integer target entries must lie in 0..16. Ratios must be positive exact rationals. A backend is a typed, versioned instrument adapter; no unspecified fallback. Physical phase command offsets, transfer and polarization calibration must be validated before any future hardware execution. The present script implements the model at local origin only; other target positions need explicit offset handling. The DSL remains a proposed specification, and a parser is not part of this commit.

Glossary: an address is a relative phase tuple, not a global coordinate; quadratic trace is F times its conjugate (or weighted vector analogue), not a field trace; concordance is a finite phase-equality predicate; optical dose is a measured exposure-dependent physical quantity, distinct from model intensity and Nyx activation density.
