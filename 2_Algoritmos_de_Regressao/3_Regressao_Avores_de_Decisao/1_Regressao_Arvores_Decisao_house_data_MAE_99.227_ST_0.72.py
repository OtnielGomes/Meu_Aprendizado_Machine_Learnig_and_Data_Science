import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Pre-processing
with open('house_multi_data.pkl', mode='rb') as f:
    X_house_training, y_house_training, X_house_test, y_house_test = \
        pickle.load(f)

# Training
decision_tree_regression = DecisionTreeRegressor()
decision_tree_regression.fit(X_house_training, y_house_training)

# Scores
training_score = decision_tree_regression.score(X_house_training,
                                                y_house_training)
test_score = decision_tree_regression.score(X_house_test, y_house_test)

# Prediction
prediction = decision_tree_regression.predict(X_house_test)

# Errors calculation
mae = mean_absolute_error(y_house_test, prediction)
mse = mean_squared_error(y_house_test, prediction)

# Pos-processing
print(f'\nTraining Score: {training_score}\nTest Score: {test_score}')
print(f'\nMAE: {mae}\nMSE: {mse}')



