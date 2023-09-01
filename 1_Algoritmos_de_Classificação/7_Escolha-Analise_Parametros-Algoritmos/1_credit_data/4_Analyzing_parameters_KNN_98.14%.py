import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Pre-processing
with open('credit_cross.pkl', mode='rb') as f:
    X_credit, y_credit = pickle.load(f)

# Testing parameters
help(KNeighborsClassifier)
parameters = {'weights': ['uniform', 'distance'],
              'n_neighbors': [3, 5, 10, 20],
              'p': [1, 2],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
grid_search = GridSearchCV(estimator=KNeighborsClassifier(),
                           param_grid=parameters)
grid_search.fit(X_credit, y_credit)
best_parameters = grid_search.best_params_
best_score = grid_search.best_score_

# Pos-processing
print(best_parameters)
print(best_score)

# {'algorithm': 'auto', 'n_neighbors': 20, 'p': 1, 'weights': 'distance'}
# 0.9814999999999999