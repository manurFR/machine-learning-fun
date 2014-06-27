## Pima Indians Diabetes Dataset

Les indiens [Pima](http://en.wikipedia.org/wiki/Pima_people) ("Akimel O'odham") vivent aux alentours de Phoenix en Arizona, aux Etats-Unis. Les femmes Pima présentent une prévalence très importante des cas de diabète (supérieure à dix fois celle du reste de la population), [attentivement suivie](http://diabetes.niddk.nih.gov/dm/pubs/pima/pathfind/pathfind.htm) depuis 1965 par les institutions de santé américaine.

![Village Pima](http://www.discoverseaz.com/Graphics/History/PimoVillage.jpg)

A la fin des années 1980, J.W. Smith *et al.* ont formaté et étudié [ces données](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes) devenues classiques dans le domaine des statistiques et du *machine learning*. Leur article initial décrit ces données et les résultats de leur étude avec un premier algorithme :
[Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2245318/), Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. -- *Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265).*

### Analyse préliminaire

Le jeu de données contient les résultats d'analyse pour 768 femmes. 
L'étude originale ayant un objectif prédictif à 5 ans, les femmes ayant déjà été diagnostiquées avant l'analyse ont été écartée du dataset. En revanche, un diabète a été détecté pour 268 d'entre elles dans les cinq ans qui suivent la visite médicale, soit un **taux d'incidence de 34,9%** (7% par an), effectivement très élevé.

En termes de *machine learning*, cela signifie que les populations des deux classes à prédire sont inégales. Il sera donc probablement plus pertinent d'utiliser des métriques de type *precision*, *recall*, et *score F1* (plutôt que la simple *accuracy*).

Les *features* (les valeurs mesurées) sont :

* *pregnancies* : nombre de grossesses
* *glucose* : concentration de glucose dans le plasma sanguin deux heures après [l'ingestion d'une dose standardisée](http://en.wikipedia.org/wiki/Glucose_tolerance_test#Standard_OGTT) (mg/dl)
* *blood pressure* : pression artérielle (diastolique) (mm Hg)
* *skin* : [épaisseur de la peau](http://www.topendsports.com/testing/skinfold-tricep.htm) au niveau du triceps (mm), mesure apparemment corrélée à la masse graisseuse
* *insulin* : quantité d'insuline dans le sang deux heures après ingestion de glucose (mm U/ml)
* *bmi* : body mass index (poids en kg / carré de la hauteur en mètres)
* *pedigree* : résultat d'une fonction assez complexe, décrite dans la publication de J.W. Smith *et al* citée plus haut, et qui traduit la densité de membres de la famile déjà diagnostiqués comme diabétiques, pondérée par la proximité du parent (les pères, mères, frères, soeurs ont plus de poids que les cousins, etc.). Valeurs comprises entre 0.078 et 2.420.
* *age* : en années

On constate que toutes ces *features* sont des nombres, dont la plupart (à part le nombre de grossesses et l'âge) sont continues.

Analyse statistique des *features*:
```>>> pima.describe()```

| &nbsp; | pregnancies | glucose    | blood pressure | skin       | insulin    | bmi        | pedigree   | age        |
|--------|-------------|------------|----------------|------------|------------|------------|------------|------------|
| mean   |  3.845052   | 120.894531 |  69.105469     | 20.536458  |  79.799479 |  31.992578 |   0.471876 |  33.240885 |
| std    |  3.369578   |  31.972618 |  19.355807     | 15.952218  | 115.244002 |   7.884160 |   0.331329 |  11.760232 |
| min    |  0.000000   |   0.000000 |   0.000000     |  0.000000  |   0.000000 |   0.000000 |   0.078000 |  21.000000 |
| 25%    |  1.000000   |  99.000000 |  62.000000     |  0.000000  |   0.000000 |  27.300000 |   0.243750 |  24.000000 |
| 50%    |  3.000000   | 117.000000 |  72.000000     | 23.000000  |  30.500000 |  32.000000 |   0.372500 |  29.000000 |
| 75%    |  6.000000   | 140.250000 |  80.000000     | 32.000000  | 127.250000 |  36.600000 |   0.626250 |  41.000000 |
| max    | 17.000000   | 199.000000 | 122.000000     | 99.000000  | 846.000000 |  67.100000 |   2.420000 |  81.000000 |

On peut voir qu'il existe des lignes avec un body mass index à zéro, ce qui signifie une donnée manquante, qu'il faudra probablement remplacer par une valeur raisonnable.
Même remarque pour l'épaisseur de la peau au triceps et la pression artérielle.
 
#### Graphes
On trace le *boxplot* de la répartition des *features* :

![Répartition des features](charts/boxplot.png)

On constate que la mesure du *glucose* est bornée à 200 mg/dl, car un patient au delà de cette valeur est considéré comme diabétique au moment du test et que ces patients ont été exclus du dataset.

Histogramme des *features* :

![Histogrammes population complète](charts/histo_global.png)

Le même histogramme, séparé entre les résultats pour les femmes non-diabétiques :
```>>> pima.groupby('diabetic').hist()```

![Histogrammes femmes non-diabétiques](charts/histo_nondiab.png)

...et pour les femmes diabétiques :

![Histogrammes femmes diabétiques](charts/histo_diab.png)

On constate que la courbe de pression artérielle est décalée vers les valeurs élevées pour les femmes diabétiques.

La courbe pour le taux de glucose est notablement différente. On peut superposer les deux histogrammes (femmes diabétiques [foncé] et non-diabétiques [clair]) :

![Histogramme Glucose comparé](charts/histo_glucose.png)

Enfin, on trace les graphes de la matrice de covariance entre les *features*, avec leurs densités sur la diagonale :

![Scatter plot features](charts/scatterplot.png)

On peut y rechercher des corrélations, comme celle assez claire entre l'épaisseur de peau au triceps et le body mass index.
On constate également (dernières ligne et colonne) qu'aucune *feature* ne présente une transition nette et donc ne permet de prédire seule la future incidence du diabéte.

> Written with [StackEdit](https://stackedit.io/).