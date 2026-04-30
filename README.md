Versus est un outil de comparaison textuelle combinant similarité sémantique et similarité lexicale pour identifier, classer et aligner automatiquement des documents. Le processus repose sur une vectorisation pondérée des phrases (TF-IDF/BM25), permettant de calculer une similarité globale entre textes (cosinus + BM25) afin de sélectionner les documents les plus proches, puis d’effectuer un alignement fin phrase à phrase via un score sémantique (cosinus entre embeddings), lexical (Jaccard et overlap) ou combiné (60 % sémantique + 40 % lexical).

L’interface guide l’utilisateur en quatre étapes — chargement, traitement (par similarité ou mots-clés), visualisation des alignements détaillés et exploration statistique — avec mise en évidence des mots communs et des différences, scores interprétables et outils de filtrage. Un module de visualisation avancé complète l’analyse (heatmap, radar, Sankey, etc.), et les résultats peuvent être exportés en CSV pour exploitation externe.

Ce développement constitue une réécriture complète de l’outil, indépendante des versions précédentes (voir les autres branches du dépôt).

Motasem Alrahabi
Avril, 2026