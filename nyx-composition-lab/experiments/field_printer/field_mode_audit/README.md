# Field-mode completeness: audit response (P1a)

## Scope and reproducibility

This is a new exact counterexample to the completeness of the four-wave optical backend, not a replacement of Nyx or a demonstration of resin curing. The earlier physical_bridge remains intact. Python 3.13.5, standard library only.

    python field_modes.py > results.json

The local Python program checked all 4913 relative phase-offset triples on eight selected addresses; 8 additional samples certify a fifth-direction witness. The proof below applies to arbitrary offsets and constant amplitudes, not just the enumerated phase settings.

## Why no raster or mechanical motion is required by the goal

DeepSeek's review (26 Nov 2026, maintainer-supplied) correctly requests optical calibration, inverse design, separate computation and metrology, and a material dose-response law. Its CPU/FPGA, NTT, Klipper/CoreXY, voxel slicer and G-code proposals are engineering options, not verified necessities. We defer FPGA pending a demonstrated compute bottleneck. Z/17Z has no multiplicative element of order 17. A modular accelerator over other suitable primes would require a separately audited exact reconstruction.

Do not turn this project into a serial positioning system: the goal is synthesis of a volumetric dose distribution by physical fields. The optical modulator is the primary programmable actuator. Sample placement or instrument alignment can use auxiliary motion. No mandatory CoreXY or G-code backend is proposed.

External physical references:
- Lin, Baker, Menon, single-exposure holographic lithography, https://doi.org/10.1038/s41467-026-73975-4
- Single-exposure holographic 3D printing via inverse-designed phase masks, https://pmc.ncbi.nlm.nih.gov/articles/PMC13464468/
- DISH, digital incoherent synthesis, https://doi.org/10.1038/s41586-026-10114-5 (distinct coherence regime).

These works establish physical fabrication by optical fields. They do not validate HFP-17 or the material-field transfer of Nyx. Raw experimental arrays have not been imported here.

## Exact obstruction

In the existing relative phase chart j=(x,y,z), the four mode exponents are 0,x,y,z. A global command adds one constant offset per mode and changes its amplitude if supported. The quadratic intensity is a sum of pair terms, each depending on at most two of x,y,z. Its alternating difference over any product of two addresses in each coordinate is therefore zero. This also holds for a sum of exposures in a linear additive dose model with unchanged propagation modes.

Choose the eight addresses {0,8}^3. Prescribe four target addresses with an even number of 8 coordinates and four keep-out addresses with an odd number. These labels specify a finite engineering requirement, not a Boolean ontology. If all targets have dose >T and all keep-out points have dose <T for a common threshold, the sum of the four target doses must exceed the sum of the four keep-out doses. The alternating identity forbids this. The same obstruction applies to weak activation >=T on targets with strict <T outside.

Thus no choice of global offsets, constant amplitudes, exposure weights, or number of exposures in this fixed four-mode linear dose model can realize this requested support. Per-address independent optimization would hide the coupling imposed by shared commands. Spatially varying transfer, nonlinear material response or moving/changing propagation modes are outside this impossibility proof.

## A fifth physically admissible direction

Use the P1a direction numerators divided by 101. Add s4=(0,-20,99)/101 and transverse polarization e4=(0,99,20)/101. Both have unit norm; their dot product is zero. Since s4=s1+s2-s3, its relative exponent is x+y-z. This is a new mode in the same local physical chart, not a new native primitive. The rational direction is a candidate instrument geometry, not a measured beam.

For the witness, use only channels 0 and 4 (independent channel switching is a proposed hardware requirement). Their normalized intensity is 2+(99/101)(zeta^(x+y-z)+zeta^(-x-y+z)). On the four target addresses, residues are 0 or 16; on the four keep-out addresses, residues are 8 or 9. In the calibrated positive real optical embedding the first cosines are positive and the second negative, as follows directly from their turn intervals. Thus threshold 2 strictly separates these eight samples. No floating-point trigonometric calculation is required.

The fifth direction removes this specific obstruction; it does not demonstrate arbitrary shape synthesis, a continuous three-dimensional feature, absence of repeated maxima, or physical cure. Other addresses are not certified. The previous P1a normalized threshold is not reused as a resin dose.

## Compiler consequence

FieldScript should compile a target envelope and a specified keep-out envelope to a common set of field modes and commands; the forward map must evaluate their joint effect. Return a certificate or a typed STOP. Do not assign independent phases to each address unless the actuator can actually provide them. Preserve rational phase relations and exact coefficients. Future spatial modulators need calibrated transfer maps and global inverse design.

P1b: record phase steps, amplitudes, polarization, and the measured intensity map under a pinned configuration. P2: record dose-response, absorption and actual material support with uncertainties. Test the entire enclosing physical volume, not merely eight samples. P3: measure the native structural fields using their corpus-defined instruments and bridges. Native capacitary overlap and Morcillien require calibrated admissible envelopes; no invented weights are introduced here.

Audit: try to break the alternating identity, the fifth-mode realization and the threshold sign certificate. QED is exact within the declared model; QEG is a conditional optical geometry; QEC for material formation remains open. This is an audit response and proposal, not an external endorsement.
