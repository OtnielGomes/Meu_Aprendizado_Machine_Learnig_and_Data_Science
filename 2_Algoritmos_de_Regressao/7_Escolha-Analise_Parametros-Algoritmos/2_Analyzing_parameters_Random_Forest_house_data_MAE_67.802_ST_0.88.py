import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
# Pre-processing

with open('house_cross_data.pkl', mode='rb') as f:
    X_house, y_house = pickle.load(f)

# Testing parameters
# help(RandomForestRegressor)
parameters = {'n_estimators': [10, 20, 30, 40, 50],
              'criterion': ['squared_error'],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10]}
scoring = make_scorer(mean_absolute_error, greater_is_better=False)
grid_search = GridSearchCV(estimator=RandomForestRegressor(),
                           param_grid=parameters, scoring=scoring, cv=5)
grid_search.fit(X_house, y_house)


best_parameters_grid = grid_search.best_params_
best_score_grid = grid_search.best_score_
best_regressor = grid_search.best_estimator_
prediction = best_regressor.predict(X_house)
mae = mean_absolute_error(y_house, prediction)

# Pos-processing

print(f'\n{best_parameters_grid}\n{best_score_grid}')
print(f'MAE: {mae}')

# {'criterion': 'squared_error', 'min_samples_leaf': 1,
# 'min_samples_split': 2, 'n_estimators': 100}
# -69072.71066256534
# MAE: 25631.676303636512