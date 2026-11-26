# De Ratione et Conversione Datorum — discipline de recherche

26 novembre 2026. Axiomatique de Nyx / HFP-17. Addendum méthodologique à Feuillet IV V2, sans réécriture des théorèmes, des données historiques ni des programmes. Source interne : *Critica Rationis Purae*, 25 novembre 2026, §§ IV, VII, XI et XII; notamment le critère de factorisation et la conservation des relations. Ce document est une règle de travail du projet, non une nouvelle primitive de Nyx.

## 1. Convertir la mesure, ne pas importer le modèle

Revenir aux observations physiques les plus proches de l'acquisition, avec provenance, état de l'échantillon, ordre, conditions instrumentales et résolution. Reconstituer les grandeurs utiles par les opérateurs natifs démontrés. Ne pas prendre une grandeur ajustée, lissée ou reconstruite par un modèle rejeté pour une observation brute. Un nombre publié peut être conservé comme trace documentaire sans être un argument admissible du calcul.

Les outils mathématiques rejetés par l'auteur sont hors de l'outillage actif de Nyx. Ne pas les réintroduire par renommage, par normalisation opportuniste, par une bibliothèque ou par analogie. Les preuves et contre-exemples motivant leur rejet restent dans la documentation. Une identité vraie dans un modèle extérieur ne démontre pas à elle seule une relation du monde dans Nyx. La critique d'une opération et sa portée restent celles des démonstrations du corpus.

## 2. Chaîne mathématique

Notons X le domaine d'observations individuelles effectivement recueillies et P une représentation extérieure déjà calculée. Pour une relation R à retrouver sur X, si R = h ∘ P, alors P(a)=P(b) implique R(a)=R(b). Par conséquent une paire ayant P(a)=P(b) et R(a)≠R(b) suffit à exclure la reconstruction par cette seule représentation.

Le contre-exemple de Critica Rationis Purae §IV, dans l'algèbre cyclique déclarée à 17 positions :
    
    e_j e_k = e_(j+k mod 17)
    P6(e6) = P6(e11) = 0
    P6(e6 e11) = P6(e0) = e0 ≠ 0

La projection perd les facteurs sans perdre leur produit observable. Elle ne transporte donc pas automatiquement la composition. Ne pas substituer un traitement de P(X) au calcul natif sur X.

Chaîne opératoire : acquisition originale → conservation des relations et des conditions de mesure → conversion démontrée dans la jauge native → composition et prédiction natives → confrontation avec observations indépendantes. Les unités instrumentales restent documentées. Une conversion non déterminée n'est pas une prédiction négative. Ne pas inventer une nouvelle primitive ou un coefficient pour forcer le passage.

## 3. Application immédiate aux verres métalliques

Lambson et al. (1986), DOI 10.1103/PhysRevB.33.2380 : rechercher les séries acoustiques individuelles, conditions de température et pression, dimensions, état et histoire de formation, informations de calibration et, si disponibles, temps de transit et signaux originaux. Les modules et rapports dérivés publiés restent des témoins documentaires, pas des entrées natives automatiques.

Haruyama et al. (2007), DOI 10.4028/www.scientific.net/MSF.539-543.2059 : remonter aux mesures de densité, de masse/volume ou aux signaux instrumentaux effectivement disponibles. Le volume libre réduit 0.0117 et la valeur approximative 0.12 du traité n'ont pas de conversion démontrée entre eux; 400/39 est seulement le rapport des nombres cités, pas un facteur de conversion. Ne pas recalibrer βv. Conserver la loi historique et les équations fondamentales sans modification; reconstituer la grandeur native depuis les mesures et la géométrie des enveloppes.

Diffraction et structure locale : conserver les intensités et informations instrumentales disponibles, avec le protocole. Les reconstructions atomistiques sont des résultats d'inférence, à documenter séparément; ne pas les confondre avec des positions directement observées.

## 4. Requalification des anciens résultats

- Comparaison utilisant un résultat de modèle extérieur sans conversion démontrée : comparaison non admissible; ne pas comptabiliser comme échec de Nyx.
- Échec d'une composition synthétique : résultat valide dans le contrat fini déclaré, utile pour écarter une configuration, non verdict sur la matière.
- Contre-exemple à l'adaptateur optique à quatre modes : défaut du dispositif mathématique étudié; ne pas généraliser à toutes les architectures ou à Nyx.
- Ruptures de contacts de PANEL-01 : témoins topologiques pour améliorer le dessin; ni fréquence de rupture physique ni mesure de résistance.
- Désaccord obtenu avec observations correctement converties et un même régime : conserver et investiguer, sans conclure par avance à la faute de la mesure ou de la loi.

L'ancienne vérification de l'identité élastique isotrope externe, dans FOLIO_IV.md et le feuillet, reste une trace d'audit de la lecture des données et ne doit pas servir de calcul actif dans Nyx. Les scripts historiques restent archivés, pas déclarés automatiquement conformes à cette discipline. Les marqueurs STOP éventuels sont des états de traitement documentaire/informatique, pas une primitive ontologique.

## 5. Production et statut

Aucun ajustement aux cibles observées. Séparer les constantes dérivées, les entrées mesurées, les décisions de conception et les tolérances/étalonnages du dispositif physique. La connaissance d'un état matériel ne doit pas être remplacée par une recette d'ajustement.

Le présent addendum ne prétend pas avoir récupéré les données brutes, ni avoir exécuté un nouveau test physique ou validé l'intégralité des scripts. Aucune loi ni aucun résultat historique n'est supprimé. L'audit d'admissibilité de chaque entrée et la reconstruction native des grandeurs matérielles constituent le travail suivant.
