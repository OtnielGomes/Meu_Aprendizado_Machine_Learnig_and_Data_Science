import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

### Pré_processamento ###
with (open('census_data.pkl', mode='rb') as f):
    X_census_treinamento, y_census_treinamento,\
    X_census_teste, y_census_teste = pickle.load(f)

### Treinamento ###
naive_census = GaussianNB()
naive_census.fit(X_census_treinamento, y_census_treinamento)

### Previsões ###
previsoes = naive_census.predict(X_census_teste)

### Pós_processamento ###
precisao_algoritimo = accuracy_score(y_census_teste, previsoes)
print(precisao_algoritimo)

cm = ConfusionMatrix(naive_census)
cm.fit(X_census_treinamento, y_census_treinamento)
cm.score(X_census_teste, y_census_teste)
cm.show()