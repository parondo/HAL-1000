# Data first — source and type audit

The maintainer's rule for this continuation: investigate measurements, specimen history, provenance, unit and type mismatches before modifying the published laws. Retain all disagreements. Do not assume a source must be wrong because it disagrees. No new coefficient adjustment, no new statistical estimator.

## Metallic glasses

Source: [Traité des Matériaux de Jauge, §§5.5–5.6](https://www.fdaa.fr/traite-des-materiaux-de-jauge/). The retrieved indexed section distinguishes the two anchor regimes and the free-volume term:

    ν = 1/4 + (3/92)(1 − δ/δico) + βv Φlibre
    βv ≈ 1.25, reported as calibrated from the two glasses.

The predictions 0.3506 and 0.3962 are inside the reported ranges [0.347,0.355] and [0.393,0.403]. The two rows outside their printed intervals are instead i-Al-Cu-Fe and W, by 1/5000 and 7/11500 respectively. These are arithmetic checks of reported values, not independent measurements. The table calls the quantities raw; the original specimen records and uncertainty definitions were not recovered. The glass rows are not independent parameter-free validations of βv.

Independent reference: [Phys. Rev. B 33, 2380 (1986)](https://doi.org/10.1103/PhysRevB.33.2380), acoustic elasticity of Pd40Ni40P20, gives room-temperature ν≈0.40 and bulk/shear ratio 4.56. [Relaxation Kinetics of Bulk Metallic Glasses below Glass Transition Temperature](https://doi.org/10.4028/www.scientific.net/MSF.539-543.2059) reports reduced free volume 0.0117 for as-quenched Pd40Ni40P20 and 0.0353 for Zr55Cu30Ni5Al10. The latter is not Vitreloy 4. The approximate 0.12 from the source corpus divided by 0.0117 gives 400/39; this does not establish a decimal error or any conversion.

Conditional exact check, with βv fixed at 5/4 and δ/δico between zero and one: the predicted ν=0.3962 requires Φlibre between 13063/143750 and 731/6250. The approximate 0.12, if treated as exact, lies above that interval. This does not resolve the input rounding or type mismatch. The same calculation for Vitreloy 4 is recorded in data_audit.json. Preserve source numbers and seek independent same-specimen elasticity, density/free-volume normalization, thermal state, and fracture/structure records. No recalibration.

## Aluminium–alumina: hierarchical composition

[Seleznev, Roy-Mayhew and Faust, Scientific Reports 13, 16334 (2023)](https://doi.org/10.1038/s41598-023-43514-y), Methods, Figs. 3, 6–8: ceramic particle-loaded preform is printed, debound and sintered, then infiltrated by molten aluminium alloy. Printed-bead ceramic loading 60%; software infill 52%, 60%, 80%, 100%. The paper itself labels resulting fractions as apparent; 31%, 36%, 48%, 60%. Products of the two nominal fractions are 39/125, 9/25, 12/25, 3/5. 31% is rounded, not an inconsistency. Published flexural moduli at 36/48/60%: 89/129/147 GPa. Constituent reference moduli in Fig. 3 are from literature, not measured on each constituent of these specimens. Do not treat these as native local f, local modulus, or exact phase volume.

Interfaces are reactive (AlN and magnesium–aluminium spinel reported). At nominal 52% infill the orthogonal and gyroid structures show approximately 1.6 times work of fracture for gyroid, with negligible reported change in modulus or strength. Keep architecture and reaction history, not just fraction. The article's Data availability states that datasets are obtainable from the corresponding author upon reasonable request. No raw specimen arrays have been acquired. The earlier global substitution is quarantined as an invalid type transfer, not a test of the native local law. Do not fit a correction to these aggregate rows.

## Dual-color volumetric printing: two interpenetrating networks

[Wang et al., Nature Communications 13, 367 (2022)](https://doi.org/10.1038/s41467-022-28013-4), Fig. 2e, Methods and Supplementary Fig. 13. Blue ~455 nm and UV ~365 nm. Acrylate reacts to both illumination bands, epoxy to UV in the reported formulation. The two interpenetrating networks are not covalently linked to each other; dose ratio, conversion, local mass/volume fractions and density of activation are distinct. Do not assign complementary fractions merely from the two light channels.

Dogbone tensile results, single-source endpoints:
| state | UV | blue |
|---|---:|---:|
| before additional postprocessing | 542 kPa | 159 kPa |
| after additional postprocessing | 201 MPa | 127 MPa |

The additional treatment combines heat and 405 nm exposure; it changes the material state. Contrasts as exact arithmetic on the printed nominal numbers: 542/159 before and 201/127 after. Do not pool these states or interpret the endpoints as pure constituents. Compressive and AFM measurements in Fig. 1 use different methods; the paper reports aggregates for parts of these figures, not interchangeable raw tensile data. No averaging or fit is performed here.

The original article identifies a [source data XLSX](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-022-28013-4/MediaObjects/41467_2022_28013_MOESM7_ESM.xlsx). Both web retrieval and container download failed on 26 November 2026. Data location is pinned, not a claim of acquisition. Retrieve specimen-level entries with methods, provenance and units before applying the native metrics. No workbook has been uploaded to this repository.

## Next data/command contract

Keep (1) raw source and measurement units, (2) specimen, constituent and local geometry, (3) processing state, (4) typed envelope, (5) transport and defect separate. For synthetic studies use explicitly declared finite profiles; for actual material states obtain measured fields rather than manufacturing a profile from a scalar observation. Only then compose profiles and assess native capacity. A mismatch is a finding, not grounds to discard a measurement or adjust a coefficient. No external raw arrays imported.
