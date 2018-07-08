import pandas as p
import numpy as np
import xgboost as xgb
from metrics import eval_and_print, render_features_importance
from preprocess import preprocess

dataset = p.read_csv('data/cars.csv')
dataset = dataset.apply(preprocess, axis='columns')

# Removes empty values from class.
dataset = dataset.loc[np.logical_not(np.isnan(dataset['price']))]

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1:].values

# Running default parameters for model. Should try for example GridSearchCV for best results.
regressor_model = xgb.XGBRegressor()

feature_names = dataset.columns.values[:-1]

scores = ('r2', 'explained_variance')
eval_and_print(regressor_model, X, Y, scores=scores, folds_nr=10)

regressor_model.fit(X, Y)
render_features_importance(regressor_model, dataset)
