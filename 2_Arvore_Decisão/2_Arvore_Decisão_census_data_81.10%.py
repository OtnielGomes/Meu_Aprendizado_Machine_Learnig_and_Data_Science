import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
### Pré_processamento ###
with open('census_data.pkl', mode='rb') as f:
    X_census_treinamento, y_census_treinamento,\
        X_census_teste, y_census_teste = pickle.load(f)

### Treinamento ###
arvore_census = DecisionTreeClassifier(criterion='entropy', random_state=0)
arvore_census.fit(X_census_treinamento, y_census_treinamento)

### Previsoes ###
previsoes = arvore_census.predict(X_census_teste)

### Pós_processamento ###
precisao_algoritimo = accuracy_score(y_census_teste, previsoes)
print(precisao_algoritimo)
print()
cm = ConfusionMatrix(arvore_census)
cm.fit(X_census_treinamento, y_census_treinamento)
cm.score(X_census_teste, y_census_teste)
cm.show()
print(classification_report(y_census_teste, previsoes))

