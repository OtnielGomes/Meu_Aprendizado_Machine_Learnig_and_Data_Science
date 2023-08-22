import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
### Pré_processing ###
with (open('credit.pkl', mode='rb') as f):
    X_credit_training, y_credit_training,\
        X_credit_test, y_credit_test = pickle.load(f)
### Training ###
knn_credit = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_credit.fit(X_credit_training, y_credit_training)

### Prediction ###
prediction = knn_credit.predict(X_credit_test)
accuracy_prediction = accuracy_score(y_credit_test, prediction)

### Pós_processing ###


def main():
    print(accuracy_prediction)
    cm = ConfusionMatrix(knn_credit)
    cm.fit(X_credit_training, y_credit_training)
    cm.score(X_credit_test,y_credit_test)
    cm.show()
    print(classification_report(y_credit_test, prediction))


if __name__ == '__main__':
    main()
