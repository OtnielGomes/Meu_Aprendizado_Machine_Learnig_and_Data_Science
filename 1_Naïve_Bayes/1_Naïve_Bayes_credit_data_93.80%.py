import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

### Pré_processamento ###
with (open('credit.pkl', mode='rb') as f):
    X_credit_training, y_credit_training,\
    X_credit_test, y_credit_test = pickle.load(f)

### Treinamento ###
naive_credit = GaussianNB()
naive_credit.fit(X_credit_training, y_credit_training)

### Previsoes ###
prediction = naive_credit.predict(X_credit_test)
acuuracy_prediction = accuracy_score(y_credit_test,prediction)

### Pós_precessamento ###


def main():
    print(acuuracy_prediction)
    cm = ConfusionMatrix(naive_credit)
    cm.fit(X_credit_training, y_credit_training)
    cm.score(X_credit_test, y_credit_test)
    cm.show()
    print(classification_report(y_credit_test, prediction))


if __name__ == '__main__':
    main()