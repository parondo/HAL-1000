# De Compositione et Proprietatibus Materiae — Feuillet IV

26 novembre 2026 — Patrick Morcillo, De Incarnatione Formarum. Continuation additive du feuillet III V2 et de PANEL-01, base [e0adc59](https://github.com/parondo/HAL-1000/commit/e0adc59f4196dd5cf357f10546a512082ca5f4b2). Notice scientifique de continuité du feuillet HTML illustré, non copie binaire de celui-ci. Aucune modification des fichiers de calcul ou de main.

## Chaîne de recherche conservée

P0 : K17 = ℚ[z]/(1+z+…+z^16), F=1+ζ^(x−a)+ζ^(y−b), I=Fσ16(F), maximum 9, ambiguïté par conjugaison. P1a : carte locale inverse X=101(u−v)/40, Z=−101(u+v)/4, Y=(101w+2Z)/20. 4913 adresses; M=1604/101, hors cible <12535/808, seuil T=25367/1616, ε<297/1616. Le cube de parité interdit une séparation stricte à quatre modes; le cinquième mode lève l'obstruction locale mais laisse 2597 adresses actives hors des quatre cibles. Ne pas confondre intensité et activation de matière. Voir [gauge revision](../gauge_revision/README.md), [optical mode audit](../field_mode_audit/README.md).

P1b/P2 : métrologie optique et réponse de cure; P3 : D, δ, f, χc et états de matière séparés; P4 : recapture native avec défaut, pas assimilation des coordonnées à la jauge. La correction de frustration est (δ*/δico)^2=(5/8)(13/16)=65/128, et (23/32)^2−65/128=9/1024. L'ancienne moyenne est écartée. Types de δ distincts.

## Composition et inversion

    (F ⊗ G)(z) = max_u min(F(u), G(z−u)), indices modulo 17.
    K(v) ⇒ H(u+v) = 1 si K(v) ≤ H(u+v), sinon H(u+v).
    ε_K(H)(u) = min_v (K(v) ⇒ H(u+v)).

min(a,b)≤c ssi a≤(b⇒c) par les deux cas b≤c et b>c. Donc δ_K(F)≤H ssi F≤ε_K(H). Une cible exacte est réalisable dans le domaine brut ssi δ_K(ε_K(H))=H, par monotonie et adjonction. Les contraintes de contact, de support et de capacité restent à vérifier. Associativité brute : maximum sur u+v+w=z des minima de trois valeurs; n'implique pas que toute composition partielle soit admissible.

| j | F | G | G′ | F⊗G | F⊗G′ |
|---|---|---|---|---|---|
|0|1|1|1|1|1|
|1|3/4|1/2|0|3/4|3/4|
|2|0|0|1/2|1/2|1/2|
|3|0|0|0|0|1/2|

V(F)={0,1,2}, V(G)={0,1}, V(G′)={0,2}. Le second résultat sort de l'union; rejet, pas écrêtage. Pour K=F, le résidu de la cible étroite H=F est 1 à 0 seulement; de la cible large H=F⊗G, il est G.

μλ(A)=(1−λ)|A|/17+λ|A|(34−|A|)/289. Pour trois adresses, capacité ≤1/4 ssi 0≤λ≤85/168. λ=1/2 est un choix de témoin synthétique, pas un ajustement de matière. μ=72/289. cμ=49/72, eμ=23/72, d△=25/289 pour la perte d'un site. L'arête rompue peut laisser le support inchangé. Le Morcillien complet requiert les centres et distances calibrés.

La loi locale du composite E_eff=E₂+f^(5/8)(E₁−E₂) donne f^5=r_E^8, r_E=(E_eff−E₂)/(E₁−E₂), E₁≠E₂. Les témoins f=t^8, r_E=t^5 sont exacts. La fraction globale et le module de flexion d'une pièce ne sont pas automatiquement les champs locaux. Voir [composition native](../native_composition/README.md) et [audit des données](../native_composition/DATA_AUDIT.md).

## Panneau multifonctionnel

V0={0,1,2}, V1={1,2,3}, V2={2,3,4}, V3={3,4,5}. Les 9 contacts structurels sont 01,02,12,13,23,24,34,35,45. Deux rails électriques séparés, un canal thermique, maintien de tous les sites, isolation électrique et zones protégées thermiques. Version redondante : triangles 012 et 345, routes thermiques 1–2–4 et 1–3–4. Retirer un contact commun de tous les canaux.

| Pannes | minimale | redondante |
|---|---:|---:|
| simples | 4/9 | 9/9 |
| doubles | 6/36 | 28/36 |

90 enregistrements (45 par architecture); ce n'est pas 90 essais physiques réussis. Six suppressions de site, trois fuites électriques, quatre fuites thermiques détectées. 68 cartes affines transportent les capacités et contacts. Pas de probabilité de panne ni de conductance physique inférée.

## Verres métalliques : contrôle sans ajustement

[Traité des matériaux de jauge, §§5.5–5.6](https://www.fdaa.fr/traite-des-materiaux-de-jauge/) : ν=1/4+(3/92)(1−r)+βvΦ, r=δ/δico. Les ancrages 1/4 et 13/46 donnent 3/92; ils ne démontrent pas à eux seuls l'interpolation ou le terme libre. βv≈1,25 est historiquement calibré sur les deux verres; ne pas le présenter comme dérivé indépendamment.

Sous βv=5/4 et 0≤r≤1, Φ doit appartenir à [(ν−1/4−3/92)/(5/4),(ν−1/4)/(5/4)].

| matériau | ν observé | prédiction historique | Φ nominale | r impliqué | Φ compatible |
|---|---|---|---|---|---|
| Vitreloy 4 | 0.3510±0.0040 | 0.3506 | 0.08 | 1227/1250 | [7819/143750,503/6250] |
| Pd40Ni40P20 | 0.3980±0.0050 | 0.3962 | 0.12 | 4187/3750 | [13063/143750,731/6250] |

Le second r sort du domaine conditionnel si les entrées approximatives sont prises exactement. Le minimum est 0,4; écart à la prédiction 19/5000. 0,4 reste dans l'intervalle observationnel imprimé. Ne pas déclarer la loi réfutée par cette seule vérification.

Source indépendante [PRB 33,2380](https://doi.org/10.1103/PhysRevB.33.2380), Pd, K/G=4.56, ν≈0.40 : l'identité isotrope externe donne ν=146/367≈0.39782, contrôle de lecture et non import dans Nyx. [Ngai et al. 2014](https://arxiv.org/abs/1311.1117) rapportent ν=0.356 (VIT4) et 0.403 (Pd), compilation distincte. [Relaxation Kinetics](https://doi.org/10.4028/www.scientific.net/MSF.539-543.2059) : volume libre réduit 0.0117 (Pd trempé), 0.0353 (Zr55Cu30Ni5Al10, pas VIT4). 0.12/0.0117=400/39 : aucun pont de normalisation démontré, pas de correction décimale. STOP sur l'identification des volumes, pas d'ajustement de βv.

Reproduction de ce diagnostic, Python standard :

```python
from fractions import Fraction as Q
a,b,beta=Q(1,4),Q(3,92),Q(5,4)
for v,phi in [(Q('0.3506'),Q('0.08')),(Q('0.3962'),Q('0.12'))]:
    lo,hi=(v-a-b)/beta,(v-a)/beta
    r=1-(v-a-beta*phi)/b
    print(lo,hi,r)
assert Q('0.12')/Q('0.0117')==Q(400,39)
k=Q('4.56')
assert (3*k-2)/(2*(3*k+1))==Q(146,367)
```

## Enquête sur les interfaces et conditions extérieures

- [Seleznev et al. 2023](https://doi.org/10.1038/s41598-023-43514-y) : à remplissage nominal identique de 52%, travail de fracture du gyroïde ≈1.6 fois celui de l'orthogonal, sans changement notable rapporté du module ou de la résistance. Interfaces réactives. Les substitutions globales naïves du feuillet III restent documentées, non ajustées.
- [Wang et al. 2022](https://doi.org/10.1038/s41467-022-28013-4) : réseaux interpénétrés UV/bleu; 542/159 kPa avant traitement complémentaire, 201/127 MPa après. Ne pas confondre dose, conversion et fraction.
- [Crespo-Miguel et al. 2025](https://doi.org/10.1038/s41467-025-56707-y) : polymères conducteurs imprimés, pas verres métalliques. Méthodes : 5,7.5,10 V sur 40 mm → 125,187.5,250 V/m; Résultats : 150,187.5,250 V/m. Conserver 25 V/m de divergence, chercher les relevés. Test couplé 187.5 V/m, 300 s, puis 1/1000 s⁻¹; perte de chemins conducteurs puis refroidissement.
- Thermodynamique de jauge V2.2, régime déclaré : Ecell=E0+Dξ³=(17/8)kBT. Pour 25,35,45 °C, Ecell/kB=101371/160,104771/160,108171/160 K. Incrément 85/4 K par 10 K instrument. ξ et E0 nécessaires; pas d'activation déduite automatiquement.

## Contrat et verdict

Cible → composition inverse → enveloppes admissibles → commande physique seulement avec pont établi → mesure O → recapture typée et défaut. Distinguer constantes démontrées, choix de conception, entrées mesurées, étalonnage instrumental et tolérances de production. Aucun coefficient nouveau ajusté à la réponse cible. La pression, conductance, histoire de traitement, interfaces et données brutes par spécimen restent à acquérir et transporter. Ne pas rejeter une donnée seulement parce qu'elle contredit une prédiction.

Relecture locale complémentaire du feuillet : 1549 contrôles exacts exécutés, dont 729 adjonctions, 729 associativités, 68 cartes (capacités et contacts), deux architectures et diagnostics ci-dessus. QED : témoin fini. QEG : graphe déclaré, pas pièce physique. QEC : ouvert. Le prototype physique HFP-17 n'est pas fabriqué; aucune validation de résistance, isolation ou chaleur réelle n'est revendiquée. Le HTML complet comporte six figures vectorielles; cette notice conserve les chaînes mathématiques lisibles sans TeX exclusif. Les scripts PANEL-01 et native_composition existants restent inchangés.
