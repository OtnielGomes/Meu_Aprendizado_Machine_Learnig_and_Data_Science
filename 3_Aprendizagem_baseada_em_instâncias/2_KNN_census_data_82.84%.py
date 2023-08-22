import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
### Pré_processing ###
with (open('census_data.pkl', mode='rb') as f):
    X_census_training, y_census_training, \
        X_census_test, y_census_test = pickle.load(f)
### Training ###
knn_census = KNeighborsClassifier(n_neighbors=15, metric='minkowski', p=2)
knn_census.fit(X_census_training, y_census_training)
### Prediction ###
prediction = knn_census.predict(X_census_test)
prediction_accuracy = accuracy_score(y_census_test, prediction)


### Pós_processing ###
def main():
    print(prediction_accuracy)
    cm = ConfusionMatrix(knn_census)
    cm.fit(X_census_training, y_census_training)
    cm.score(X_census_test, y_census_test)
    cm.show()
    print(classification_report(y_census_test, prediction))


if __name__ == '__main__':
    main()

