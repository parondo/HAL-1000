# Follow-up audit brief

External review received from DeepSeek on P0 (26 Nov 2026, maintainer supplied text): documentation assessed positively; Python implementation not inspected; requested glossary, explicit language status, granular logs, runtime version, and measurable stage gates. The review did not execute the code or validate physical transfer. The earlier "alternative histogram" was a local implementation cross-check, not an earlier rejected external audit. There is no documented dispute or rejected audit.

This follow-up explicitly corrects scope: HFP-17 is a physical printer R&D project. P0 remains a finite algebraic witness; P1a adds optical forward/inverse realization with calibrated units, vector polarization and a dose-selection certificate on finite samples. External audit remains requested.

Please verify:
- Direction and polarization normalization, transversality, inverse map, local periodicity.
- Polynomial recurrence and positivity, unequal pair lower bound, vector maximum and strict error bound; look for counterexamples.
- Recapture formula, calibration assumptions and whether phase conjugation ambiguity is resolved.
- Separation of reported experimental observations, synthetic model outputs, and actual HFP17 measurements.
- Continuous/off-grid behavior, repeat maxima, exposure and cure threshold, attenuation, material characterization; identify which empirical tests are required.
- The full Nyx physical/structural bridge must preserve admissibility and typed defects; do not identify optical calibration with H_perp or material-field equivalence.

Reproduce with Python 3.13.5, standard library. Run bridge.py and python -m unittest -v, and generate address_details.jsonl using dump_details.py. Report PASS/CORRECTION/STOP with exact counterexamples or proofs. The P0 FieldScript example remains a design notation; see FIELD_LANGUAGE.md for grammar. No compiler for hardware is implemented.
