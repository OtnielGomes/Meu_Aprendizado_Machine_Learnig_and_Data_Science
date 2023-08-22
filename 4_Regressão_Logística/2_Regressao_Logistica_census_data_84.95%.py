import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

### Pré_processing ###
with open('census_data.pkl', mode='rb') as f:
    X_census_training, y_census_training,\
        X_census_test, y_census_test = pickle.load(f)

### Training ###
logist_census = LogisticRegression(random_state=0)
logist_census.fit(X_census_training, y_census_training)

### Prediction ###
prediction = logist_census.predict(X_census_test)
prediction_accuracy = accuracy_score(y_census_test, prediction)


### Pós_processing ###
def main():
    print(prediction_accuracy)
    cm = ConfusionMatrix(logist_census)
    cm.fit(X_census_training, y_census_training)
    cm.score(X_census_test, y_census_test)
    cm.show()
    print(classification_report(y_census_test, prediction))


if __name__ == '__main__':
    main()

