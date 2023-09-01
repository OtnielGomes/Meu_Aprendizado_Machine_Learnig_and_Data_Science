import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

### Pré_processing ###
with open('credit.pkl', mode='rb') as f:
    X_credit_training, y_credit_training,\
        X_credit_test, y_credit_test = pickle.load(f)

### Training ###
logist_credit = LogisticRegression(random_state=0)
logist_credit.fit(X_credit_training, y_credit_training)

### Prediction ###
prediction = logist_credit.predict(X_credit_test)
prediction_accuracy = accuracy_score(y_credit_test, prediction)


### Pós_processing ###
def main():
    print(prediction_accuracy)
    cm = ConfusionMatrix(logist_credit)
    cm.fit(X_credit_training, y_credit_training)
    cm.score(X_credit_test, y_credit_test)
    cm.show()
    print(classification_report(y_credit_test, prediction))


if __name__ == '__main__':
    main()


