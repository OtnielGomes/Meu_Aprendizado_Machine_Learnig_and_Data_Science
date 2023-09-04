import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Pre-processing
with open('house_multi_data.pkl', mode='rb') as f:
    X_house_training, y_house_training, X_house_test, y_house_test = \
        pickle.load(f)

# Training
random_forest_regression = RandomForestRegressor(n_estimators=100)
random_forest_regression.fit(X_house_training, y_house_training)

# Scores
training_score = random_forest_regression.score(X_house_training,
                                                y_house_training)
test_score = random_forest_regression.score(X_house_test, y_house_test)

# Prediction
prediction = random_forest_regression.predict(X_house_test)

# Errors calculation
mae = mean_absolute_error(y_house_test, prediction)
mse = mean_squared_error(y_house_test, prediction)

# Pos-processing
print(f'\nTraining Score: {training_score}\nTest Score: {test_score}')
print(f'\nMAE: {mae}\nMSE: {mse}')


# {'criterion': 'squared_error', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
# -69072.71066256534
# MAE: 25631.676303636512