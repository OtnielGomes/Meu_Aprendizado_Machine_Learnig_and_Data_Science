import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Pre-processing
with open('credit_cross.pkl', mode='rb') as f:
    X_credit, y_credit = pickle.load(f)


# Testing parameters
help(RandomForestClassifier)
parameters = {'criterion': ['gini', 'entropy'],
              'n_estimators': [10, 40, 100, 150],
              'min_samples_leaf': [1, 2, 5, 10],
              'min_samples_split': [2, 5, 10]}
grid_search = GridSearchCV(estimator=RandomForestClassifier(),
                           param_grid=parameters)
grid_search.fit(X_credit, y_credit)
best_parameters = grid_search.best_params_
best_score = grid_search.best_score_

# Pos-processing
print(best_parameters)
print(best_score)

# {'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
# 0.9875