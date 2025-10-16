# Clustering Report

## 1. Compréhension des méthodes kmeans et agglémoratif (3 à 4 pages)

### a. la liste des hyperparamètres testés et pourquoi

### b. pour 3 exemples : donnez des graphiques visuels du résultat de chaque méthode et les valeurs des hyperparamètres correspondant. Précisez l’algorithme mis en place pour les obtenir.

### c. Analysez les résultats présentés

## 2. Points faibles des méthodes kmeans et agglomératif. Illustrez les points faibles de chaque méthode de manière visuelle sur 1 ou 2 jeux de données (1 à 2 pages)

## 3. Compréhension des méthodes DBSCAN et HDBSCAN (3 à 4 pages)

### a. la liste des hyperparamètres testés et pourquoi

### b. pour 3 exemples : donnez des graphiques visuels du résultat de chaque méthode et les valeurs des hyperparamètres correspondant. Précisez l’algorithme mis en place pour les obtenir.

### c. Analysez les résultats présentés

## 4. Points faibles des méthodes DBSCAN et HDBSCAN. Illustrez les points faibles de chaque méthode de manière visuelle sur 1 ou 2 jeux de données (1 à 2 pages)

## 5. Le lien vers votre dépot git (acessible à vos 3 enseignants)

```
https://github.com/YacineSteeve/insa-5a-clustering
```






````


1)

POUR KMEANS : 

-> Expliquer pourquoi on fait varier le nombre de clusters, etc...
-> Tester la méthode du coude
-> Tester la méthode qui maximise le coefficient Silhouette
-> Visualiser les résultats et les paramètres associés
 

POUR AGGLOMÉRATIF:

-> Expliquer pourquoi on fait varier le nombre de clusters, etc...
-> Tester la méthode qui maximise le coefficient Silhouette
-> Visualiser les résultats et les paramètres associés


2)

Montrer les limites de KMEANS et AGGLO (par ex sur le cluster Bananes ou sur des clusters avec des fortes différences de densité)


3)


POUR DBSCAN : 

-> Expliquer pourquoi on fait varier le nombre min de voisins, epsilon, etc…
-> Montrer pourquoi ça marche sur certains clusters qui ne marchaient pas avec les deux précédents
-> Visualiser les résultats et les paramètres associés


POUR HDBSCAN : 

-> Expliquer pourquoi on fait varier le nombre min de voisins, epsilon, MinPts etc…
-> Montrer pourquoi ça marche mieux sur certains clusters qui ne marchaient pas avec les trois précédents
-> Visualiser les résultats et les paramètres associés


4)

-> Montrer pourquoi HDBSCAN et DBSCAN ne fonctionnent pas sur certains jeux de données (trouver lesquels à partir du cours)

5)


-> Faire un algo qui permet de trouver le bon nombre de clusters (switcher entre les méthodes et essayer de trouver celle qui donne les meilleurs métriques selon son type)
```` 

