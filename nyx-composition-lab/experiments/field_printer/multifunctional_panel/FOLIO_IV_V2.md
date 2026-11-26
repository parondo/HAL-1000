# Feuillet IV V2 — De Compositione et Proprietatibus Materiae

26 novembre 2026. **Version documentaire active**. La V1 et les calculs historiques restent conservés. [Notice des chaînes mathématiques V1](FOLIO_IV.md) (lire avec l'erratum documentaire V2 ci-dessous).

- [Feuillet illustré HTML (7 figures SVG, équations, preuves, sources)](Incarnatio_04_De_Compositione_et_Proprietatibus_Materiae_V2.html)
- [Code WordPress intégral, annoté](Incarnatio_04_De_Compositione_et_Proprietatibus_Materiae_V2_WordPress.txt)
- [Vérificateur Python autonome](Incarnatio_04_V2_verification.py)
- [Rapport JSON intégral](Incarnatio_04_V2_rapport.json)

## Audit V2 des verres métalliques

Lambson et al. (1986), DOI 10.1103/PhysRevB.33.2380 : Pd40Ni40P20, K/G=4.56, ν≈0.40, étude acoustique selon température et pression. Comparaison documentaire : 0.360 (citation secondaire à vérifier), 0.403 (Ngai 2014), 0.398±0.005 (traité), prédiction historique 0.3962. Vitreloy 4 : 0.351±0.004 (traité), 0.356 (compilation), 0.3506 (prédiction). Aucun regroupement ou lissage.

La formule du traité est conservée : ν=1/4+(3/92)(1−δ/δico)+βvΦlibre. βv≈1.25 reste une calibration historique, pas un nouveau coefficient. Haruyama et al. (2007), DOI 10.4028/www.scientific.net/MSF.539-543.2059 : x_reduit=0.0117 (Pd), 0.0353 pour Zr55Cu30Ni5Al10 (pas VIT4). 0.12/0.0117=400/39 : définition/normalisation à vérifier, pas correction décimale. Sous les hypothèses de l'ancien contrôle, voir les bornes et le défaut conservés dans le feuillet. Pression : de Hey et al. (1997), dx/dp=−0.0025 GPa⁻¹ dans leur modèle; à 0.75 GPa, Δx=−3/1600. Ne pas assimiler x, ΔV/V et Φlibre. Hosokawa et al. (2019), organisation locale issue de diffraction et reconstruction : voisinages icosaédriques préférentiels autour de Ni, plus déformés autour de Pd et P. Ce ne sont pas des champs natifs directement mesurés. Sources et limites détaillées dans le HTML.

## Reproduction du feuillet V2

```sh
python3 Incarnatio_04_V2_verification.py > /tmp/folio_iv_replay.json
diff -u Incarnatio_04_V2_rapport.json /tmp/folio_iv_replay.json
```

Rejoué dans le bac à sable le 26 novembre 2026 : 1553/1553 contrôles PASS, rapport identique octet pour octet. V1 : 1549/1549, également rejoué à l'identique. Ces contrôles portent sur l'algèbre finie et l'arithmétique des valeurs publiées, pas sur la fabrication. Aucun nouveau paramètre ajusté. Pas de modification des théorèmes. Les tests précédents de PANEL-01, composition native, mode optique et autres modules restent dans leurs dossiers, avec leurs procédures et leurs rapports. Ce rejeu ne constitue pas un nouveau test de tous les scripts du dépôt.

STOP : données brutes par spécimen, état thermique et mécanique, transport de la pression et des propriétés électriques, définition de Φlibre, réalisation physique. QED : témoins finis; QEG : contacts déclarés; QEC physique ouvert.


## Addendum de méthode — 26 novembre 2026

[De Ratione et Conversione Datorum](DATA_CONVERSION_POLICY.md) précise la requalification des comparaisons historiques et le retour aux observations originales. Il complète le feuillet illustré sans modifier ses chaînes ni ses résultats archivés.
