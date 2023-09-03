import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Pre-processing
with open('house_multi_data.pkl', mode='rb') as f:
    X_house_training, y_house_training, X_house_test, y_house_test = \
        pickle.load(f)
# Staggering
X_scaler = StandardScaler()
y_scaler = StandardScaler()
# X
X_house_training_staggered = X_scaler.fit_transform(X_house_training)
X_house_test_staggered = X_scaler.transform(X_house_test)
# y
y_house_training_staggered = y_scaler.fit_transform(y_house_training.
                                                    reshape(-1, 1))
y_house_test_staggered = y_scaler.transform(y_house_test.reshape(-1, 1))

# Training
neural_network_regression = MLPRegressor(hidden_layer_sizes=(9, 9),
                                         activation='relu',
                                         solver='adam',
                                         batch_size=64,
                                         max_iter=5000,
                                         tol=1e-6,
                                         verbose=True)
neural_network_regression.fit(X_house_training_staggered,
                              y_house_training_staggered.ravel())
# Prediction
prediction = neural_network_regression.predict(X_house_test_staggered)
# Scores
training_score = neural_network_regression.score(X_house_training_staggered,
                                                 y_house_training_staggered)
test_score = neural_network_regression.score(X_house_test_staggered,
                                             y_house_test_staggered)

# Errors calculation
y_house_test_inverse = y_scaler.inverse_transform(y_house_test_staggered)
prediction_inverse = y_scaler.inverse_transform(prediction.reshape(-1, 1))

mae = mean_absolute_error(y_house_test_inverse, prediction_inverse)
mse = mean_squared_error(y_house_test_inverse, prediction_inverse)

# Pós-processing
print(f'\nTraining Score: {training_score}\nTest Score: {test_score}')
print(f'\nMAE: {mae}\nMSE: {mse}')

