import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

advertising = pd.read_csv("advertising.csv")

from sklearn.model_selection import train_test_split

feature_cols = ['TV', 'Radio', "Newspaper"]

X = advertising[feature_cols]
y = advertising.Sales

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=123)

from sklearn.linear_model import LinearRegression
from sklearn import metrics


model = LinearRegression(fit_intercept=True)

model.fit(Xtrain.values, ytrain)


filename = "model/model.pkl"
pickle.dump(model, open(filename, "wb"))

loaded_model = pickle.load(open(filename, "rb"))


print(loaded_model.predict([[4, 300, 500]]))


