import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Pre-processing
with open('house_multi_data.pkl', mode='rb') as f:
    X_house_training, y_house_training, X_house_test, y_house_test = \
        pickle.load(f)

poly = PolynomialFeatures(degree=2)
X_house_training_poly = poly.fit_transform(X_house_training)
X_house_test_poly = poly.transform(X_house_test)

# Training
linear_regression_house = LinearRegression()
linear_regression_house.fit(X_house_training_poly, y_house_training)

# Scores
training_score = linear_regression_house.score(X_house_training_poly,
                                               y_house_training)
test_score = linear_regression_house.score(X_house_test_poly, y_house_test)

# Prediction
prediction = linear_regression_house.predict(X_house_test_poly)

# Errors calculation
mae = mean_absolute_error(y_house_test, prediction)
mse = mean_squared_error(y_house_test, prediction)
# Pos-processing
print(f'\nTraining Score: {training_score}\nTest Score: {test_score}')
print(f'\nMAE: {mae}\nMSE: {mse}')

