# Physical field printer — optical adapter, revision P1a

## Purpose

HFP-17 is intended to become a physical field-composition printer. P0 was an algebraic milestone, not the scope of the whole project. This revision provides a concrete forward/inverse optical phase map and a finite dose-selection certificate. It uses published fabrication observations as external physical inputs. It does not claim that a HFP-17 machine has already printed a specimen.

## Reproduce

Python 3.13.5 was used locally; standard library only. From this directory:

    python bridge.py
    python -m unittest -v
    python dump_details.py > address_details.jsonl

Five unit-test groups passed locally; the detailed journal contains 4,913 address records. The result file records Python version and exact fractions. The original P0 commit and results are preserved. The new tests are an independent implementation cross-check, not an external reviewer endorsement.

## Physical calibration and exact forward/inverse map

Let lambda_m=lambda_0/n, with lambda_0=405 nm and n=1.613 (published SU-8 nominal values, not new measurements). The optical path step is lambda_0/17, and the geometric path-difference step in homogeneous material is lambda_m/17. Neither is a printable feature size.

Choose four equal-amplitude coherent plane-wave directions s_i and unit polarizations e_i given in bridge.py. The direction numerators divided by 101 are (0,0,101), (20,0,99), (-20,0,99), (0,20,99). The polarization numerators are (0,101,0) for the first three, and (0,99,-20) for the fourth. All are unit vectors and each e_i is orthogonal to its own s_i. These choices are not a claim about a published experimental apparatus. They define an ideal homogeneous isotropic optical interface in O, with the physical interpretation of phases supplied by calibration, not an ontology of Nyx.

For position r in units of lambda_m, relative phase turns are u_i=(s_i-s_0) dot r. Choose representative turns (u,v,w)=(j1,j2,j3)/17. The inverse is

    X = 101(u-v)/40
    Z = -101(u+v)/4
    Y = (101w+2Z)/20

This gives a nonsingular three-dimensional local phase chart. Every one of the 4,913 addresses is checked by a separate forward calculation. The map is periodic in relative turns, so an address is not a unique global location; the physical envelope and local reference must be specified. Absolute coordinates are only the instrument adapter, not a native metric primitive.

## Vector interference and threshold certificate

Let zeta be the physical realization of a 1/17-turn phase, with the chosen positive orientation. At the sampled addresses the normalized electric field is E=sum e_i zeta^j_i, with j0=0. Its squared electric-field norm I=sum_{i,k}(e_i dot e_k) zeta^(j_i-j_k) is an exact cyclotomic polynomial. This is proportional to local absorption in an ideal isotropic linear absorbing medium with fixed coefficient; the general vector Poynting flux, attenuation and nonlinear resin response require further modeling.

The real cyclotomic polynomial for t=zeta^a+zeta^-a, a nonzero, is

    p(t)=t^8+t^7-7t^6-6t^5+15t^4+10t^3-10t^2-4t+1.

The coefficients of p(s+15/8) are all strictly positive (see results.json), hence no root is >=15/8. The recurrence for z^k+z^-k independently recovers p. Each unequal phase pair therefore loses strictly more than (e_i dot e_k)/8 from concordance. A nonconstant quadruple has at least three unequal unordered pairs and all pair weights are >=99/101.

The vector maximum is M=1604/101, and every other sampled address has I < M-297/808 = 12535/808. Thus T=25367/1616 separates the unique maximum sample from the rest even with strictly bounded absolute normalized intensity error <297/1616. This is a conservative certificate, not a claim of an optimal gate. A single physical phase cell must be selected by the envelope: periodically repeated maxima and continuous neighborhoods are not eliminated by this finite certificate. It does not guarantee a continuous gap, suppress side lobes, or produce an arbitrary geometry. Translation of a target is a relative phase offset, not a hardware trajectory.

For an ideal constant exposure, dose is proportional to intensity times exposure time. To turn this finite certificate into a manufacturing test, measure material cure threshold, attenuation and calibrated amplitude; verify an exposure exists with the worst-case target dose above threshold and every out-of-target dose below threshold. Off-grid validation is mandatory. The target dose and intensity gate here are model-derived normalized values, not an experimentally calibrated resin dose or the Nyx density threshold.

## Recapture

Block or separate other channels. For a calibrated equal-amplitude two-channel test with a 17-step reference phase, I_k=2+zeta^(a+k)+zeta^(-a-k). Then

    (1/17) sum_{k=0}^{16} I_k zeta^(-k) = zeta^a.

This follows from the finite root-of-unity sum. All 17 phase states are reconstructed exactly in the model. This resolves the original conjugate ambiguity when signed phase steps are calibrated. With real measurements one must retain uncertainty and all compatible phase states; no rounding to a convenient address without an error bound. This is a measurement protocol, not an already recorded interferogram.

## Published physical data

Lin, Baker, Menon, Nature Communications 17, 7175 (2026), DOI https://doi.org/10.1038/s41467-026-73975-4 : Fig. 1/Methods (405 nm, index 1.613, nominal 720 um depth and ~6 um walls); Fig. 5a (approximate 80 um measured shift over 450 um air propagation); Fig. 6 (approximate capillary depths 169,233,243 um at 1,10,20 s; reported effective modulus 5.7 GPa, extracted from compression curve). These are published observations or reported derived quantities, not independent HFP17 measurements. Decimal/rounded reported values are represented exactly as reported, not asserted to be exact physical quantities.

The rational import gives 8/45 -> 10, 1/120 -> 1, 27/80 -> 15 mod 17. The last ratio compares capillary endpoint with structure depth, not a proof of blocked-volume fraction. No force law, frustration, composition or chirality is inferred from these alone.

Raw supplementary arrays could not be fetched in this session. The source states that experimental data and masks are in its supplement. The public DISH repository was inspected at commit 1fcede0635097e359a68e70cfa3da5b52ac4b437: it contains a beam profile PNG and a target TIFF, but binary retrieval failed. The target TIFF is design input, not a measurement. DISH is incoherent synthesis across projections, not the coherent four-wave model. Do not mix them.

Sources: https://doi.org/10.1038/s41586-026-10114-5 ; https://github.com/sugar10w/DISH ; https://doi.org/10.5281/zenodo.17905914 ; https://doi.org/10.1126/sciadv.adf6182 ; phase-stepping measurement background https://doi.org/10.1186/s41476-016-0016-6 (verify exact article identifier before formal citation). Native material target fields: https://www.fdaa.fr/traite-des-materiaux-de-jauge/ ; engineering source https://www.fdaa.fr/2026/09/18/342/ (full page fetch failed; no unverified constitutive law imported).

## Validation gates

P1a: exact optical adapter and finite phase-selection/recapture certificates, implemented. Physical realization in ideal homogeneous medium is conditional on the stated optical assumptions. P1b: calibrated laboratory phase steps, amplitudes, polarization, and actual propagation; compare raw interferograms to the model. P2: independent dose/response sweep and measurement of printed support; no fabricated resin parameters. P3: measure structural target fields delta, f, chi_c through appropriate calibrated instruments; no identification of optical phase with material chirality. P4: full native capacities, contacts, Morcillien and physical closure with declared defects once the measurements and typing are available. No statistical estimator imported into the active Nyx core.

QED: finite algebraic proof and tests. QEG: explicit conditional optical chart in O, not a demonstrated transfer into H_perp. QEC: physical HFP17 realization not yet independently measured. The physical aim is affirmative; validation is a separate status.
