import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

### Pré_processamento ###
with open('census_data.pkl', mode='rb') as f:
    X_census_treinamento, y_census_treinamento,\
        X_census_teste, y_census_teste = pickle.load(f)

### Treinamento ###
random_forest_census = RandomForestClassifier(
    n_estimators=100, criterion='entropy', random_state=0)
random_forest_census.fit(X_census_treinamento, y_census_treinamento)

### Previsoes ###
previsoes = random_forest_census.predict(X_census_teste)

### Pós_processamento ###
precisao_algoritimo = accuracy_score(y_census_teste, previsoes)
print(precisao_algoritimo)
print()
cm = ConfusionMatrix(random_forest_census)
cm.fit(X_census_treinamento, y_census_treinamento)
cm.score(X_census_teste, y_census_teste)
cm.show()
print(classification_report(y_census_teste, previsoes))

