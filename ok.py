
from main import log_reg

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('dataa.csv')
X = dataset.loc[ : , dataset.columns != 's']
y = dataset['s']
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

log_reg_result = log_reg(X_train, y_train, 0.5, 100, X_test)
print(log_reg_result)