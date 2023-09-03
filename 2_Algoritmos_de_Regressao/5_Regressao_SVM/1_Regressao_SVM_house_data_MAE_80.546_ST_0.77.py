import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
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
svm_regression = SVR(kernel='rbf',
                     tol=1e-3,
                     C=7,
                     degree=4,
                     verbose=True)
svm_regression.fit(X_house_training_staggered,
                   y_house_training_staggered.ravel())

# Scores
training_score = svm_regression.score(X_house_training_staggered,
                                      y_house_training_staggered)
test_score = svm_regression.score(X_house_test_staggered,
                                  y_house_test_staggered)
# Prediction
prediction = svm_regression.predict(X_house_test_staggered)

# Errors calculation
y_house_test_inverse = y_scaler.inverse_transform(y_house_test_staggered)
prediction_inverse = y_scaler.inverse_transform(prediction.reshape(-1, 1))

mae = mean_absolute_error(y_house_test_inverse, prediction_inverse)
mse = mean_squared_error(y_house_test_inverse, prediction_inverse)

# Pos-processing
print(f'\nTraining Score: {training_score}\nTeste Score: {test_score}')
print(f'\nMAE: {mae}\nMSE: {mse}')
