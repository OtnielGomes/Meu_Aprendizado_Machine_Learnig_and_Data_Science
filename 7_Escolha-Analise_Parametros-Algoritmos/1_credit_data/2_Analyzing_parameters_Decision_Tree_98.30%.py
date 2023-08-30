import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Pre-processing
with open('credit_cross.pkl', mode='rb') as f:
    X_credit, y_credit = pickle.load(f)

# Testing parameters
help(DecisionTreeClassifier)
parameters = {'criterion': ['entropy', 'gini'],
              'splitter': ['best', 'random'],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10]}
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(),
                           param_grid=parameters)
grid_search.fit(X_credit, y_credit)
best_parameters = grid_search.best_params_
best_score = grid_search.best_score_

# Pos-processing
print(best_parameters)
print(best_score)

# {'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 5,
# 'splitter': 'best'}
# 0.983