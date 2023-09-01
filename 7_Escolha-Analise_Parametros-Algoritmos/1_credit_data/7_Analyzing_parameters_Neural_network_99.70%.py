import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

# Pre-processing
with open('credit_cross.pkl', mode='rb') as f:
    X_credit, y_credit = pickle.load(f)

# Testing parameters
help(MLPClassifier)
parameters = {'hidden_layer_sizes': [10],
              'batch_size': [8, 10, 16, 32, 56, 64, 128, 256],
              'solver': ['adam'],
              'activation': ['relu'],
              'max_iter': [30000],
              'tol': [0.00000001],
              'verbose': [True]}
grid_search = GridSearchCV(estimator=MLPClassifier(), param_grid=parameters)
grid_search.fit(X_credit, y_credit)
best_parameters = grid_search.best_params_
best_score = grid_search.best_score_

# Pos-processing
print(best_parameters)
print(best_score)

# {'activation': 'relu', 'batch_size': 56, 'hidden_layer_sizes': 10,
# 'max_iter': 30000, 'solver': 'adam', 'tol': 1e-08, 'verbose': True}
# 0.998