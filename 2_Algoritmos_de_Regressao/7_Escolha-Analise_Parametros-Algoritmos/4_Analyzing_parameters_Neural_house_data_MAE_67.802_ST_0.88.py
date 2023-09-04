import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, make_scorer

# Pre-processing
with open('house_cross_data.pkl', mode='rb') as f:
    X_house, y_house = pickle.load(f)

# Staggering
X_scaler = StandardScaler()
y_scaler = StandardScaler()
# X
X_house_staggered = X_scaler.fit_transform(X_house)
# y
y_house_staggered = y_scaler.fit_transform(y_house.reshape(-1, 1))

# Parameters
parameters = {'hidden_layer_sizes': [100, 10, (9, 9), (100, 100)],
              'activation': ['relu'],
              'solver': ['adam'],
              'batch_size': [8, 24, 48, 64, 128, 256],
              'max_iter': [30000],
              'tol': [1e-7],
              'verbose': [True]}
scoring = make_scorer(mean_absolute_error, greater_is_better=False)

grid_search = GridSearchCV(estimator=MLPRegressor(),
                           param_grid=parameters,
                           scoring=scoring)
grid_search.fit(X_house_staggered, y_house_staggered.ravel())

best_parameters = grid_search.best_params_
best_score = grid_search.best_score_

best_regressor = grid_search.best_estimator_
prediction = best_regressor.predict(X_house_staggered)
y_house_inverse = y_scaler.inverse_transform(y_house_staggered)
prediction_inverse = y_scaler.inverse_transform(prediction.reshape(-1, 1))

mae = mean_absolute_error(y_house_inverse, prediction_inverse)
# Pos-processing
print(f'\n{best_parameters}\n{best_score}')
print(f'\nMAE: {mae}')

# {'activation': 'relu', 'batch_size': 8, 'hidden_layer_sizes': (100, 100),
# 'max_iter': 30000, 'solver': 'adam', 'tol': 1e-07, 'verbose': True}
# -0.1995274496751612
#
# MAE: 59618.350114423454