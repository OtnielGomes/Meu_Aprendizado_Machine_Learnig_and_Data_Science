import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
### Pré_processing ###
with open('credit.pkl', mode='rb') as f:
    X_credit_training, y_credit_training, X_credit_test, y_credit_test =\
        pickle.load(f)

### Training ###
# versoes do kernel svm: 'linear', 'poly', 'rbf', 'sigmoid'
svm_credit = SVC(kernel='rbf', random_state=0, C=7)
svm_credit.fit(X_credit_training, y_credit_training)

### Prediction ###
prediction = svm_credit.predict(X_credit_test)
prediction_accuracy = accuracy_score(y_credit_test, prediction)


### Pós_processing ###
def main():
    print(prediction_accuracy)
    cm = ConfusionMatrix(svm_credit)
    cm.fit(X_credit_training, y_credit_training)
    cm.score(X_credit_test, y_credit_test)
    cm.show()
    print(classification_report(y_credit_test, prediction))


if __name__ == '__main__':
    main()

