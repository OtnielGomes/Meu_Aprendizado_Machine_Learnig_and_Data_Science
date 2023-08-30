import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Pre-processing
with open('credit_cross.pkl', mode='rb') as f:
    X_credit, y_credit = pickle.load(f)

# Testing parameters
help(SVC)
parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'C': [1.0, 1.5, 2.0],
              'tol': [1e-3, 0.00001, 0.000001]}
grid_search = GridSearchCV(estimator=SVC(),
                           param_grid=parameters)
grid_search.fit(X_credit, y_credit)
best_parameters = grid_search.best_params_
best_score = grid_search.best_score_

# Pos-processing
print(best_parameters)
print(best_score)

# {'C': 1.5, 'kernel': 'rbf', 'tol': 0.001}
# 0.9829999999999999