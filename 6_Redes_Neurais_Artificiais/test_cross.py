import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
### Pré_processing ###
with open('credit_cross.pkl', mode='rb') as f:
    X_credit, y_credit = pickle.load(f)

### Training ###
neural_network_credit = MLPClassifier(max_iter=20000,
                                      verbose=True,
                                      batch_size=1024,
                                      tol=0.0000001,
                                      solver='adam',
                                      activation='relu',
                                      hidden_layer_sizes=(2, 2))
neural_network_credit.fit(X_credit, y_credit)
#0.00699733
#0.00425171
#0.00279744
#0.00164565
#0.00076080
### Prediction ###
prediction = neural_network_credit.predict(X_credit)
prediction_accuracy = accuracy_score(y_credit, prediction)


### Pós_processing ###
def main():
    print(prediction_accuracy)
    cm = ConfusionMatrix(neural_network_credit)
    cm.fit(X_credit, y_credit)
    cm.score(X_credit, y_credit)
    cm.show()
    print(classification_report(y_credit, prediction))
    help(MLPClassifier)


if __name__ == '__main__':
    main()

