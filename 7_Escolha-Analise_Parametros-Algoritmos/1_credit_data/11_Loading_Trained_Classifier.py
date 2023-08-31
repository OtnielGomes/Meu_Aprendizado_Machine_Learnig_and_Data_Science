import pickle

# Pre-Processing
decision_tree = pickle.load(open('tree_classifier.sav', mode='rb'))
svm = pickle.load(open('svm_classifier.sav', mode='rb'))
neural_network = pickle.load(open('neural_classifier.sav', mode='rb'))

# Testing

with open('credit_cross.pkl', mode='rb') as f:
    X_credit, y_credit = pickle.load(f)
new_register = X_credit[1999]
print(new_register)
print(new_register.shape)
new_register = new_register.reshape(1, -1)
print(new_register)
# Prediction

prediction_tree = decision_tree.predict(new_register)
prediction_svm = svm.predict(new_register)
prediction_neural = neural_network.predict(new_register)

# Pos-Processing

print(f'Score Decision Tree : {prediction_tree}')
print(f'Score Svm: {prediction_svm}')
print(f'Score Neural Network: {prediction_neural}')

