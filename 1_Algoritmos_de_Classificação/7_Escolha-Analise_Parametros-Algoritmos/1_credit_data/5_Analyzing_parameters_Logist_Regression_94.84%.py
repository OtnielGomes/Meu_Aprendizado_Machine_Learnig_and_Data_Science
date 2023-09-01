import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Pre-processing
with open('credit_cross.pkl', mode='rb') as f:
    X_credit, y_credit = pickle.load(f)

# Testing parameters
help(LogisticRegression)
parameters = {'tol': [0.0001, 0.00001, 0.000001],
              'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky',
                         'sag', 'saga'],
              'C': [1.0, 1.5, 2.0]}
grid_search = GridSearchCV(estimator=LogisticRegression(),
                           param_grid=parameters)
grid_search.fit(X_credit, y_credit)
best_parameters = grid_search.best_params_
best_score = grid_search.best_score_

# Pos-processing
print(best_parameters)
print(best_score)

# {'C': 1.0, 'solver': 'lbfgs', 'tol': 0.0001}
# 0.9484999999999999