# Experimental reference and import audit

Crespo-Miguel, Lucarini, Garzon-Hernandez et al., *In-silico platform for the multifunctional design of 3D printed conductive components*, Nature Communications 16, 1359 (2025), https://doi.org/10.1038/s41467-025-56707-y. Results Fig. 1, Methods, Data availability. The paper uses conductive PLA/carbon-black filament, different orientations, local adhesion and voids, and simultaneous thermo-electro-mechanical experiments. This is a physical reference for coupled failure, not the HFP-17 manufacturing method. Their continuous constitutive models and PSO calibration are not imported.

The fully coupled test imposes 187.5 V/m for five minutes, then applies tensile strain at 0.001/s with field maintained. Reported loss of conductive pathways after necking precedes cooling. Our topological deletion models the loss of declared paths, not the temperature or stress at which this occurs. We do not predict the paper's cooling curve from a graph.

**Data-first finding:** Results mention 150, 187.5, 250 V/m; Methods give 5, 7.5, 10 V at electrode separation 40 mm, yielding exactly 125, 187.5, 250 V/m. Difference 25 V/m at first setting. Preserve both, verify source records before choosing either. No claim that this is necessarily a typo.

Thermo-mechanical setting: 25, 35, 45 °C. With the corpus relation from *Thermodynamique de Jauge* v2.2 theorem 3,
    Ecell = E0 + D ξ³ = (17/8) kB T,
the instrument temperatures in kelvins give Ecell/kB = 101371/160, 104771/160, 108171/160 K. A 10 K increment changes this normalized energy by 85/4 K. This is exact source-law arithmetic conditional on its regime, not a measurement of D, not a fitted coefficient, not a validation of the constitutive law. The local ξ and E0 must be known to evaluate D. Do not equate reported temperature with density or activation predicate. Prior thermodynamic erratum remains applicable.

[Published source workbook](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-025-56707-y/MediaObjects/41467_2025_56707_MOESM3_ESM.xlsx): retrieval failed (cache miss) during this session. Authors state source data and supplement are provided; raw data upon request. We transcribe the published test settings only, no specimen-level arrays or digitized curves. No empirical prediction score can yet be computed.

Other native sources: [Engineering of gauge materials](https://www.fdaa.fr/2026/09/18/342/), [Gauge thermodynamics](https://www.fdaa.fr/thermodynamique-de-jauge/), [Les Pieds dans l'Eau](https://www.fdaa.fr/les-pieds-dans-leau/), [Morcillien statistics](https://www.fdaa.fr/411-2/), Scala Caelestis V4 PV49 and previous native_composition. Indexed sections verified, direct full-page access to several FDAA pages failed; no claim of full source freeze.

Physical validation requires specimen-level phase/geometry, temperature, pressure, stress, electrical resistance, time history and fracture data. The paper gives a direct challenge to the simulator: preserve directional microstructure and processing history; predict where pathways disappear before comparing to measured histories. Keep any discrepancy; do not adjust the corpus to fit.
