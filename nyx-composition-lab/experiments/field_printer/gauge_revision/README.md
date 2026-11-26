# HFP-17 — native gauge revision (research folio III)

**Physical objective unchanged:** programmable fields form material, not a serial mechanical toolpath. This revision responds to the maintainer's metric/type correction. It is a research contract for the next simulation, not a claim that the device has fabricated material. Earlier commits remain unchanged.

## Read and reproduce

- [P0 phase algebra](../README.md), [P1a optical chart](../physical_bridge/README.md), [field-mode obstruction](../field_mode_audit/README.md).
- [Typed contract](contract.json), [source register](SOURCES.md), [audit history and next simulation](AUDIT.md).
- `empirical_archive/` preserves previous optical and material calculations, full numerical observations as transcribed, signed discrepancies and code. These are historical diagnostics in O or source-table arithmetic; **do not import them as active Nyx metrics or physical transfer laws**.
- Python 3.13.5, standard library. Run from this directory:

    python verify_contract.py
    cd empirical_archive
    python -m unittest -v
    python empirical_check.py
    python material_contract.py

Local replay: 34/34 named contract/arithmetic checks, 4/4 archival optical test groups, 5/5 previous bridge test groups. The material output reproduced byte-for-byte. The 34 checks are not independent experimental observations, QEC or proof of a universal material model. The JSON reports include exact fractions and archived source hashes.

## Native contract

Use admissible envelopes, typed contacts, capacities, intrinsic clock, composition and calibrated Morcillien. Keep observation O, gauge reconstruction and instrument command distinct. Source laws already exist in the materials corpus. Do not invent a material law to make an optical model work.

Binary local composition: f^5 = r_E^8, r_E = (E_target-E2)/(E1-E2), under the declared local material regime. The prior global aluminium/alumina substitution is a negative control of an invalid import; not a native empirical test. The optical count 2601/4913 is a count, not a capacity. Retain all residuals and provenance.

The corrected Frustration Controlee addendum replaces the earlier arithmetic-mean resonance by the squared relation 65/128. The old 23/32 is documentary only. The engineering article's density D and local composition f are distinct fields; no silent identification. Its continuous variation/regularization, resolution and chiral stability prescriptions are historical source material to migrate selectively, not automatically executable operations.

The native comparison is c_mu(A,B)=mu(intersection)/mu(union), e_mu=1-c_mu, d_triangle=mu(symmetric difference), with admissible union and positive denominator. D_G^13=d^16 is a nonnegative rational power comparison, not a license to insert an uncalibrated distance. Full rho_G requires calibrated S3 centers and theta, not a cosine from an arbitrary phase label. No product capacity on J17^3 is inferred.

A tiny finite PV49 witness uses lambda=1 on J17, A={0,1}, B={0}. Union capacity 64/289 is below 1/4. Overlap 33/64, normalized defect 31/64, symmetric-difference capacity 33/289. It verifies the operators on their declared finite domain, not the printer. No new material simulation has been run.

## Next step

Use the existing material laws and empirical source tables to specify a typed local material target and preservation envelope. Build a calibrated process transfer from common field commands to the target states, retain fibres and all defects; recapture the measured states before evaluating native metrics. No conversion by guesswork. FieldScript remains a proposed language, not an implemented compiler. The validation gates are recorded in the contract; a missing bridge returns STOP, never a zero or false physical state. No machine-control code.
