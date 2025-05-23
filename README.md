# Benchmark adaptatif pour le calcul de l'attention

### Objectif

Ce projet vise à calculer l'attention, en optimisant le choix des paramètres de parallélisation pour minimiser le temps de calcul. On propose deux méthodes pour choisir ces paramètres en faisant moins d'essais qu'une grille : un algorithme de bandit multibras et un algorithme d'optimisation bayésienne. 

### Comment utiliser le code ?

1. Installer les dépendances nécessaires : `pip install -r requirements.txt`

2. Cythoniser la fonction de calcul de l'attention reposant sur cython : `python setup.py build_ext --inplace`

3. Exécuter les algorithmes pour optimiser les paramètres : `2. python benchmark.py --method='bayes'` et `2. python benchmark.py --method='bandit'`

4. Calculer les statistiques de comparaison des deux algorithmes : `python compare.py`
