import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
### Pré_processing ###
with open('credit.pkl', mode='rb') as f:
    X_credit_training, y_credit_training, X_credit_test, y_credit_test =\
        pickle.load(f)

### Training ###
neural_network_credit = MLPClassifier(max_iter=3000,
                                      verbose=True,
                                      tol=0.0000100,
                                      solver='adam',
                                      activation='relu',
                                      hidden_layer_sizes=(2, 2))
neural_network_credit.fit(X_credit_training, y_credit_training)

### Prediction ###
prediction = neural_network_credit.predict(X_credit_test)
prediction_accuracy = accuracy_score(y_credit_test, prediction)


### Pós_processing ###
def main():
    print(prediction_accuracy)
    cm = ConfusionMatrix(neural_network_credit)
    cm.fit(X_credit_training, y_credit_training)
    cm.score(X_credit_test, y_credit_test)
    cm.show()
    print(classification_report(y_credit_test, prediction))


if __name__ == '__main__':
    main()

