import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
### Pré_processamento ###
with open('credit.pkl', mode='rb') as f:
    X_credit_training, y_credit_training,\
    X_credit_test, y_credit_test = pickle.load(f)

### Treinamento ###
arvore_credit = DecisionTreeClassifier(criterion='entropy', random_state=0)
arvore_credit.fit(X_credit_training, y_credit_training)

### Previsoes ###
prediction = arvore_credit.predict(X_credit_test)
accuracy_prediction = accuracy_score(y_credit_test, prediction)

### Pós_processamento ###


def main():
    print(accuracy_prediction)
    print()
    cm = ConfusionMatrix(arvore_credit)
    cm.fit(X_credit_training, y_credit_training)
    cm.score(X_credit_test, y_credit_test)
    cm.show()
    print(classification_report(y_credit_test, prediction))


if __name__ == '__main__':
    main()




