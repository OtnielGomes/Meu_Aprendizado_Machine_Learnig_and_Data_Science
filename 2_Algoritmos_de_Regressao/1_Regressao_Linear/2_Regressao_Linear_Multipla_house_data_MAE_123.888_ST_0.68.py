import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Pre-processing
with open('house_multi_data.pkl', mode='rb') as f:
    X_house_training, y_house_training, X_house_test, y_house_test = \
        pickle.load(f)

# Training
linear_regression_house = LinearRegression()
linear_regression_house.fit(X_house_training, y_house_training)

# b0/b1
b0 = linear_regression_house.intercept_
b1 = linear_regression_house.coef_

# Scores
training_score = linear_regression_house.score(X_house_training,
                                               y_house_training)
test_score = linear_regression_house.score(X_house_test, y_house_test)

# Prediction
prediction = linear_regression_house.predict(X_house_test)

# Errors calculation
mae = mean_absolute_error(y_house_test, prediction)
mse = mean_squared_error(y_house_test, prediction)

# Pos-processing
print(f'\nb0: {b0}\nb1: {b1}')
print(f'\nTraining Score: {training_score}\nTest Score: {test_score}')
print(f'\nMAE: {mae}\nMSE: {mse}')
