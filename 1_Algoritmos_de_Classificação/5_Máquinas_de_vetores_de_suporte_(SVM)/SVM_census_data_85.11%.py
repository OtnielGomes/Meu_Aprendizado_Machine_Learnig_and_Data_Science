import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

### Pré_processing ###
with open('census_data.pkl', mode='rb') as f:
    X_census_training, y_census_training, X_census_test, y_census_test = \
        pickle.load(f)

### Training ###
# versoes do kernel svm: 'linear', 'poly', 'rbf', 'sigmoid'
#linear: 85.07%, poly: 82.96%, sigmoid: 82.16%, rbf: 84.93%
svm_census = SVC(kernel='linear', random_state=0, C=7)
svm_census.fit(X_census_training, y_census_training)

### Prediction ###
prediction = svm_census.predict(X_census_test)
prediction_accuracy = accuracy_score(y_census_test, prediction)


### Pós_processing ###
def main():
    print(prediction_accuracy)
    cm = ConfusionMatrix(svm_census)
    cm.fit(X_census_training, y_census_training)
    cm.score(X_census_test, y_census_test)
    cm.show()
    print(classification_report(y_census_test, prediction))


if __name__ == '__main__':
    main()

