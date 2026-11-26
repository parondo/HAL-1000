# PANEL-01 — custom multifunctional material

26 November 2026. HFP-17 / HAL-1000 research. Continuation of [native composition](../native_composition/README.md), [gauge revision](../gauge_revision/README.md), and [optical mode audit](../field_mode_audit/README.md). **Physical aim: field-formed material with structural, electrical and thermal functions in the same part.** Present experiment: finite synthetic design and fault-injection tests; no physical panel has been fabricated.

## Reproduce

Python 3, standard library only. Run in this directory:

    python panel.py > replay.json
    python -m unittest -v
    python empirical.py > empirical_replay.json
    diff -u results.json replay.json
    diff -u empirical.json empirical_replay.json

The Python model and two independent test groups were executed locally. An alternative connected-component implementation checked every one of the 90 contact-removal records. A separate JS connectivity calculation also reproduced the four crash counts. No fitting, Gaussian statistics or imported numerical optimizer. Full results include each failed configuration, not just totals.

## Typed architecture

Six address labels in a declared J17 chart; not coordinates or a physical length. Four overlapping local envelopes:

    V0={0,1,2}; V1={1,2,3}; V2={2,3,4}; V3={3,4,5}.

Local unit-profile composition: F=G=1 on {0,1}; H=F⊗G=1 on {0,1,2}. Translate H into each local envelope. Shared addresses agree in the structural profile. This is a declared synthetic sewing witness, not a proof of physical material-interface transport. The test only checks local equality and contact admissibility; full PV25/PV50 physical sewing is not established.

The PV49 witness uses λ=1/2, local budget 1/4: μ(Vi)=72/289 < 1/4. The union of six addresses would exceed the local budget; no global capacity is inferred or sum of capacities used. All nine structural edges are taken from the declared native contact set S=±{1,2,3,4,5,8} in J17, restricted to the union of the four cliques. The chart and role assignment are explicit design choices, not a new Nyx axiom.

Six simultaneous requirements:
- retain all six sites and structural connectivity;
- a conductive route from 0 to 2;
- a separate return route from 3 to 5;
- no reachable electrical connection between the two rails;
- a heat-transfer route from 1 to 4;
- no thermal path from 1 to protected sites 0 and 5.

Electrical and thermal roles are declared on contacts, not inferred from structural connectivity or scalar material fraction. A missing edge is absence from the declared functional graph, not proof of perfect physical insulation. These are topological requirements only. The physical laws of the materials corpus are retained; specimen-dependent constitutive mappings are not invented.

Compare two channel assignments on exactly the same structural graph and same local capacities. Sparse: supply 0–1–2, return 3–4–5, heat 1–2–4. Redundant: supply triangle 0,1,2; return triangle 3,4,5; heat two routes 1–2–4 and 1–3–4. An exhaustive search in these prescribed channel candidates gives minimum 3,3,4 functional edges for resistance to any one own-channel edge loss. This is not a global optimum over arbitrary material forms.

## Crash tests (finite, deterministic)

| Fault | Sparse | Redundant |
|---|---:|---:|
| Single shared contact removal | 4/9 | 9/9 |
| Double shared contact removal | 6/36 | 28/36 |

Each structural-contact deletion is propagated into all functional channels. No probability is assigned to the fault cases. For the redundant design six site deletions all fail at least one requirement (structure requires all six sites, and the deleted sites are ports). Three injected cross-rail links all violate electrical isolation. Four injected thermal links to protected sites all violate thermal keep-out. These are separate fault classes, not simultaneous leak-plus-break tests. Reachability is not a load or conductivity calculation.

The 68 affine maps j→aj+b, a in {1,4,13,16}, preserve local capacities, declared S contacts, structural connectivity and designated route reachability when all ports are transported. The full crash records were independently cross-checked, not all crash records under every affine map.

If one address in V0 is lost, the local PV49 comparison gives cμ=49/72, eμ=23/72, μ(symmetric difference)=25/289. A scalar capacity alone cannot detect an edge break at unchanged support; preserve contact-level failure information. Full Morcillien still requires calibrated native centers and angles.

## Design contract / FieldScript

    MATERIAL_TARGET panel
    COMPOSE local_envelopes
    REQUIRE STRUCTURAL_CONTINUITY
    REQUIRE ELECTRICAL_ROUTES supply, return
    PRESERVE ELECTRICAL_SEPARATION
    REQUIRE THERMAL_ROUTE source, sink
    PRESERVE THERMAL_KEEPOUT
    VERIFY SHARED_CONTACT_FAILURES 1, 2
    VERIFY LOCAL_CAPACITY, TRANSPORT, DEFECTS
    MAP_TO_COMMON_FIELD_COMMANDS ONLY_IF_CALIBRATED

This is proposed readable syntax, not a parser or executable machine controller. A graph success does not imply a mechanically strong, electrically safe or thermally adequate panel. The design is a research coupon, not a construction or mains-wiring plan.

## Next empirical gates

1. Preserve raw provenance, specimen, processing stage, measurement units, and uncertainty.
2. Recover local material states (D, δ, f, χc) and interfaces from data in O using typed recapture. Supply calibrated pressure and thermal/electrical constitutive maps. No automatic import of continuum fitting.
3. Link material state to common optical/acoustic field commands; evaluate actual envelope and keep-out defects, including off-grid physical observations.
4. Predict independent held-out measured histories before comparing. Run isolated mechanical, thermal and electrical tests, then coupled loading and deliberate defect injection. Physical crash tests require a qualified lab and appropriate load, insulation and fire-safety criteria.
5. Keep all mismatches, separate data corrections from source-law changes. QED: finite witness; QEG: declared contact realization only; QEC: physical validation open.

See [EMPIRICAL.md](EMPIRICAL.md) and empirical.json for the external test settings and the source discrepancy.


## Feuillet IV V2 — documentation active

[Consulter la V2 et ses preuves, figures, sources et tests reproductibles](FOLIO_IV_V2.md). La V1 reste une trace historique; aucun script antérieur n'est modifié.
