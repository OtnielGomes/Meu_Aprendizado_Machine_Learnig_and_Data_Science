import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

### Pré_processamento ###
with open('census_data.pkl', mode='rb') as f:
    X_census_training, y_census_training,\
        X_census_test, y_census_test = pickle.load(f)

### Treinamento ###
random_forest_census = RandomForestClassifier(
    n_estimators=100, criterion='entropy', random_state=0)
random_forest_census.fit(X_census_training, y_census_training)

### Previsoes ###
prediction = random_forest_census.predict(X_census_test)
prediction_accuracy = accuracy_score(y_census_test, prediction)


### Pós_processamento ###
def main():
    print(prediction_accuracy)
    print()
    cm = ConfusionMatrix(random_forest_census)
    cm.fit(X_census_training, y_census_training)
    cm.score(X_census_test, y_census_test)
    cm.show()
    print(classification_report(y_census_test, prediction))


if __name__ == '__main__':
    main()

