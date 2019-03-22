# Isolation Forest Implementation


The goal of this project is to implement the original [Isolation Forest](IsolationForestPaper.pdf) algorithm by Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou.  (A later version of this work is also available: [Isolation-based Anomaly Detection](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.673.5779&rep=rep1&type=pdf).) There are two general approaches to anomaly detection: 

1. model what normal looks like and then look for nonnormal observations
2. focus on the anomalies, which are few and different. This is the interesting and relatively-new approach taken by the authors of isolation forests.

The isolation forest algorithm is original and beautiful in its simplicity; and also seems to work very well, with a few known weaknesses. The academic paper is extremely readable so you should start there.

## Data sets

For this project, we'll use three data sets:

* [Kaggle credit card fraud competition data set](https://www.kaggle.com/mlg-ulb/creditcardfraud); download, unzip to get `creditcard.csv`

* Get cancer data into `cancer.csv` by executing [savecancer.csv](https://github.com/parrt/msds689/blob/master/projects/iforest/savecancer.py) that I provide.

* [http.zip](https://github.com/parrt/msds689/blob/master/projects/iforest/http.zip); download, unzip to get `http.csv`.

These files are not that large, but a pure Python solution for isolation forest takes too long on the whole file: 2.5 minutes on `creditcard.csv` and 5 minutes on `http.csv`.  (My hybrid python/C solution takes about two seconds. ha!)

My code assumes the data files are in the same directory as the code.


## Scoring results

Using [score.py](https://github.com/parrt/msds689/blob/master/projects/iforest/score.py), here is a sample run:

```
Running noise=False improved=False
INFO creditcard.csv fit time 0.23s
INFO creditcard.csv 18804 total nodes in 200 trees
INFO creditcard.csv score time 14.54s
SUCCESS creditcard.csv 200 trees at desired TPR 80.0% getting FPR 0.0300%

INFO http.csv fit time 0.28s
INFO http.csv 22430 total nodes in 300 trees
INFO http.csv score time 23.08s
SUCCESS http.csv 300 trees at desired TPR 99.0% getting FPR 0.0053%

INFO cancer.csv fit time 0.08s
INFO cancer.csv 8204 total nodes in 1000 trees
INFO cancer.csv score time 0.73s
SUCCESS cancer.csv 1000 trees at desired TPR 75.0% getting FPR 0.2857%
```

Due to the subsampling of the original data said and the inherent random nature of isolation forest, your results will differ even from run to run.  I'm hoping that the variance is not so high that valid programs fail the scoring, but let me know.

The indicated required score values were set using my machine and my implementation. Then I gave a range above that that are still considered to be valid.
