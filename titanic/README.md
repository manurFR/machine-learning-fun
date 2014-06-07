Etude du dataset Titanic
========================

### Présentation
Ce jeu de données est proposé par [Kaggle.com](http://www.kaggle.com/c/titanic-gettingStarted) comme projet d'initiation.
Les données rassemblent le nom et certaines caractéristiques des 1309 passagers du Titanic, divisées en deux ensembles :
- les données d'entraînement ([train.csv](train.csv)), comprenant 891 passagers, avec pour chacun d'entre eux un flag (0 ou 1) indiquant s'il a survécu au naufrage ;
- les données de test ([test.csv](test.csv)), comprenant les 418 passagers restant, pour lesquels l'objectif est de prédire s'ils ont survécu ou non.

Les participants au concours soumettent leurs prédictions au site Kaggle.com, qui les évalue sur 50% (non révélés) des données de test. A l'issue du concours, le classement final sera établi sur le score obtenu pour les 50% restant.

Les données fournies pour chaque passager sont les suivantes :  **Classe** du voyageur (1ère, 2ème ou 3ème), **Nom** complet, **Sexe**, **Age**, Nombre de **conjoints, frères et soeurs** à bord, Nombre de **parents et enfants** à bord, Référence du **billet** d'embarquement, **Tarif** du billet d'embarquement, Référence de la **cabine**, Port d'**embarquement** (Cherbourg, Southampton ou Queenstown (Irlande)).

### Première analyse des données
> [01_first_steps.py](01_first_steps.py)

Un passage préliminaire sur les données d'apprentissage fait apparaître que 38,4% des passagers de ce fichier ont survécu.
Cependant, le taux de survie est nettement plus contrasté si l'on divise la population selon le sexe et selon le prix déboursé comme le ticket :

| Prix du billet 	| < £10 	| £10 <= prix < £20 	| £20 <= prix < £30 	| £30 <= prix 	|
|----------------	|-------	|-------------------	|-------------------	|-------------	|
| Hommes         	| 10,6% 	| 18,8%             	| 24,1%             	| 33,6%       	|
| Femmes         	| 59,4% 	| 73,1%             	| 68,4%             	| 86,1%       	|

### Modèle ad-hoc
Compte-tenu de cette analyse assez rapide et aisée, il est possible de mettre en place un modèle de prédiction ad-hoc sans faire intervenir un algorithme d'apprentissage. (Ce modèle est proposé par le tutoriel associé au concours sur le site de Kaggle.com).
A ces deux variables discriminantes (sexe et prix du billet), on ajoute la classe de la cabine.

#### Processus
- Rassembler les passagers en 24 catégories (2 sexes x 3 classes x 4 intervalles de prix) ;
- Pour chaque catégorie, calculer le taux de survie dans le jeu de données d'entraînement ;
- Pour chaque passager du jeu de test, déterminer dans quelle catégorie il ou elle se trouve, et prédire sa survie si le taux de survie de cette catégorie est >= 50%.

#### Difficultés
- Les prix des billets sont simplifiés en 4 intervalles de prix ; il faut transformer le prix pour chaque passager en l'index de l'intervalle correspondant (exemple : un tarif de £12 sera dans le deuxième intervalle, entre £10 et £20).
- Certaines catégories ne correspondent à aucun passager du jeu de données d'apprentissage ; le taux de survie est alors placé arbitrairement à 0%. Après tout, le taux de survie global pour tous les passagers (38,4%) est inférieur à 50%.
- Le prix du billet (*fare*) n'est pas renseigné pour tous les passagers. Dans ce cas, on fait le choix d'utiliser la classe pour approximer l'intervalle de prix :

| Classe     | 1ère   | 2ème   | 3ème |
|------------|--------|--------|------|
| Intervalle approx. | £20-30 | £10-20 | <£10 |

#### Score
Le score utilisé dans ce concours est l'*Accuracy*. Celle-ci peut se représenter :

$$
Accuracy = \frac{Nombre\ de\ prédictions\ correctes}{Nombre\ total\ de\ prédictions}
$$




> Written with [StackEdit](https://stackedit.io/).