import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

### Pré_processamento ###
with open('credit.pkl', mode='rb') as f:
    X_credit_training, y_credit_training,\
        X_credit_test, y_credit_test = pickle.load(f)

### Treinamento ###
random_forest_credit = RandomForestClassifier(
    n_estimators=40, criterion='entropy', random_state=0)
random_forest_credit.fit(X_credit_training, y_credit_training)

### Previsoes ###
prediction = random_forest_credit.predict(X_credit_test)
accuracy_prediction = accuracy_score(y_credit_test, prediction)

### Pós_processamento ###


def main():
    print(accuracy_prediction)
    print()
    cm = ConfusionMatrix(random_forest_credit)
    cm.fit(X_credit_training, y_credit_training)
    cm.score(X_credit_test, y_credit_test)
    cm.show()
    print(classification_report(y_credit_test, prediction))


if __name__ == '__main__':
    main()