import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
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
parameters = {'kernel': ['rbf'],
              'degree': [3, 2, 4, 5, 10],
              'tol': [1e-3, 1e-6],
              'C': [1.0, 2.0, 3.0, 5.0],
              'verbose': [True]}
scoring = make_scorer(mean_absolute_error, greater_is_better=False)
grid_search = GridSearchCV(estimator=SVR(),
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

# {'C': 5.0, 'degree': 3, 'kernel': 'rbf', 'tol': 1e-06, 'verbose': True}
# -0.21986589186274239
#
# MAE: 62624.682570028024