import numpy as np
import matplotlib.pyplot as plt
from knn import KNN
from naiv_bay import naivBayes
from svm import SVM
import pandas as pd
from ucimlrepo import fetch_ucirepo
# ----------------------------------  Result of Feature Selected --------------------------------------
All = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
Best_3 = [1, 3, 12]
Best_4 = [1, 2, 3, 6]
Best_5 = [1, 2, 3, 6, 7]
Best_6 = [1, 2, 3, 4, 6, 7]
Best_7 = [1, 2, 3, 4, 6, 7, 20]
Best_8 = [1, 2, 3, 4, 6, 7, 12, 20]
Best_9 = [1, 2, 3, 4, 6, 7, 12, 13, 20]
Best_10 = [1, 2, 3, 4, 6, 7, 8, 12, 13, 20]
Best_11 = [1, 2, 3, 4, 5, 6, 7, 9, 12, 13, 20]
Best_12 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 20]
Best_13 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 15, 20]
Best_14 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 15, 17, 20]
Best_15 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 15, 16, 17, 20]
Best_16 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 15, 16, 17, 19, 20]
Best_17 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20]
Best_18 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19, 20]

# ----------------------------  Start Experiment  --------------------------------

# the experiment could choose any list of selected feature above
# we show some of the result here, the full result is shown in the paper
print('Results of KNN classification before feature selection:')
KNN(All)
print('Results of KNN classification after feature selection:')
KNN(Best_11)
print('Results of naive bayes classification before feature selection:')
naivBayes(All)
print('Results of naive bayes classification after feature selection:')
naivBayes(Best_17)
print('Results of support vector machine classification before feature selection:')
SVM(All)
print('Results of support vector machine classification after feature selection:')
SVM(Best_12)


