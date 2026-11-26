# HFP-17 — native material composition and inverse envelope

**26 November 2026 — continuation of research folio III. Physical fabrication by fields remains the objective.** This new finite witness is not a measured material or a physical printer run.

Read [the previous gauge revision](../gauge_revision/README.md), [data provenance audit](DATA_AUDIT.md), [the optical bridge](../physical_bridge/README.md) and [field-mode audit](../field_mode_audit/README.md). Nothing in these earlier chains is deleted or overwritten. The author requests investigation of data provenance and typing before any modification of source laws; no coefficient is fitted here.

## Reproduction

Python 3.13.5, standard library only. Run from this directory:

    python composition.py > replay.json
    python data_audit.py > data_replay.json
    diff -u results.json replay.json
    diff -u data_audit.json data_replay.json

The two Python scripts were executed locally. The finite restricted exhaustive test checked 729 triples for the adjunction and 729 for associativity. The declared witnesses, including rejected compositions, are in results.json. This is not an exhaustive audit over every rational profile or every physical geometry.

## 1. Source operation, not a scalar mixture

Native references:
- [Codex Existentiae](https://www.fdaa.fr/le-codex/), Appendix B.
- [Composition morphologique](https://www.fdaa.fr/2026/09/18/composition-morphologique/), §§2–4.
- [Entre Ici et le Centre](https://www.fdaa.fr/2026/11/22/entre-ici-et-le-centre/), §10.2, corrected residual.
- [L'Enfance de l'Art](https://www.fdaa.fr/lenfance-de-lart/), typed containment and capacity budget.
- Scala Caelestis V4, PV49 capacity family and declared contact graph S. Earlier gauge_revision/SOURCES.md records the corpus source access limits. Indexed excerpts, not a complete frozen source version, were available for the web pages.

The declared additive finite chart J17 uses residues 0..16. Profiles take rational values in [0,1]; zero outside declared positive support is a boundary convention for this test, not the ontological assertion that everything outside a gauge is false.

    (F ⊗ G)(z) = max over u ∈ J17 of min(F(u), G(z − u))
    All address subtraction is modulo 17.

This is the finite sup–min operator of the source. It is not intensity addition or a weighted arithmetic mean. Positive-support containment, the threshold preservation test, an explicit finite contact-clique contract and union capacity are checked separately; the raw algebraic result is not automatically admitted. The clique check is a sufficient local contract chosen for this witness, not a new universal Nyx axiom.

**No physical chart identification is implied.** No Euclidean coordinates, statistical estimates, spatial interpolation or material fitting are used in this experiment.

## 2. Exact witnesses and admissible envelope

| Address | F | G | G′ | F ⊗ G | F ⊗ G′ |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 1 | 1 | 1 |
| 1 | 3/4 | 1/2 | 0 | 3/4 | 3/4 |
| 2 | 0 | 0 | 1/2 | 1/2 | 1/2 |
| 3 | 0 | 0 | 0 | 0 | 1/2 |

Other addresses have zero in this finite chart. Envelopes: V(F)={0,1,2}, V(G)={0,1}, V(G′)={0,2}. V(F) is an explicitly prescribed envelope, not inferred from positive support. Activation threshold 1/2 is a synthetic test value, not Σ*. The PV49 family is:

    μλ(A) = (1−λ)|A|/17 + λ|A|(34−|A|)/289.

The local union budget is 1/4 (declared test). For size three the capacity is (51+42λ)/289, so the exact permitted range is 0 ≤ λ ≤ 85/168. At λ=1/2 the union is 72/289 < 1/4. This rational choice is a witness in the range, not a calibration of a material. At λ=1 the same composition fails the budget (93/289). For both G and G′ the support capacity is 49/289 at λ=1/2.

F ⊗ G preserves the active sites, stays in the union and its three addresses satisfy the contact contract. F ⊗ G′ has positive support at address 3, outside the union; it is rejected, not clipped. Equal counts and capacities do not imply equal composition. This also shows why pointwise upper bounds by the maximum of the two input profiles cannot be applied to an unrestricted raw dilation: the admissibility conditions must be checked first.

## 3. Inverse design through corrected residuation

For a fixed kernel K, a candidate F and target upper profile H, define the source residual:

    K(v) ⇒ H(u+v) = 1          if K(v) ≤ H(u+v)
                    H(u+v)    otherwise

    εK(H)(u) = min over v ∈ J17 of [K(v) ⇒ H(u+v)].

Proof: min(a,b) ≤ c iff a ≤ (b ⇒ c), by the two cases b ≤ c and b > c. Thus δK(F) ≤ H iff each min(F(u),K(v)) ≤ H(u+v), iff F(u) ≤ εK(H)(u) for each u. No negation-as-falsity or uncontrolled clipping. This is the corrected finite order adjunction, not the superseded min/max proof.

Consequently εK(H) is the greatest profile whose dilation does not exceed H. The equality target is realizable by an unconstrained profile iff δK(εK(H)) = H. Necessity follows by monotonicity and the adjunction; sufficiency takes the residual itself. Additional typed constraints may still exclude that witness; a greatest upper-bound profile is not automatically a valid material or command.

With K=F from the table:
- tight target H=F: the maximal complementary profile has only address 0 at density 1;
- wide target H=F⊗G: the residual is exactly G;
- both reconstructed targets pass equality in the finite calculation.

Associativity of raw composition follows by expanding the two nested maxima into the maximum over all triples of the minimum of their three values, with the same sum of addresses. This does **not** make every partial typed composition associative or available without checking its envelopes at each step. The 729 finite triple checks are a computational witness of these identities, not the proof itself.

## 4. Native support comparison

For the requested positive support {0,1} and the wide candidate {0,1,2}, use λ=1/2:

    cμ = 49/72
    eμ = 23/72
    d△ = 25/289

The comparison uses admissible union 72/289, not coordinate distance or a Gaussian score. This is a finite capacitary witness, not a full Morcillien: calibrated centers, gauge distance and relative angle are not yet provided. The density profiles and envelope choices are synthetic. Do not equate optical phases with material states or the set of 4913 optical addresses with native capacity.

## 5. FieldScript consequence

    MATERIAL_TARGET H
    FIXED_CONSTITUENT K
    SOLVE MAXIMAL_COMPLEMENT BY RESIDUATION
    CHECK TARGET_EQUALITY
    CHECK CONTACTS, CONTAINMENT, CAPACITY, ACTIVATION
    KEEP DEFECTS
    MAP_TO_FIELD_COMMAND ONLY WITH TYPED PROCESS_TRANSFER

This is readable design notation, not an implemented parser or hardware instruction. For real printing, bind independently observed material profiles, preserve the individual constituents, interfaces and processing history, then establish the calibrated command-to-material transfer. Source law f⁵=rE⁸ remains a typed local relation; it cannot substitute for the full morphology. Full three-field (frustration, composition, chirality) material recapture is still needed for a physical prediction. No claim that these finite witnesses validate the physical laws.

QED: finite algebraic identities with proof. QEG: declared finite chart only, not a fabricated geometry. QEC: physical specimen comparison pending. See DATA_AUDIT.md for acquired published values, source access status and unresolved transport.
