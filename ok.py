
from main import log_reg

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from LogisticRegression import *

dataset = pd.read_csv('dataa.csv')
X = dataset.loc[ : , dataset.columns != 's']
y = dataset['s']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


logReg = LogisticRegression(200000,0.6)
logReg.fit(X_train,y_train)
logReg.predictor(X_test)
print(logReg.score(X_test,y_test))