import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

### Pré_processamento ###
with open('credit.pkl', mode='rb') as f:
    X_credit_treinamento, y_credit_treinamento,\
        X_credit_teste, y_credit_teste = pickle.load(f)

### Treinamento ###
random_forest_credit = RandomForestClassifier(
    n_estimators=40, criterion='entropy', random_state=0)
random_forest_credit.fit(X_credit_treinamento, y_credit_treinamento)

### Previsoes ###
previsoes = random_forest_credit.predict(X_credit_teste)

### Pós_processamento ###
precisao_algoritimo = accuracy_score(y_credit_teste, previsoes)
print(precisao_algoritimo)
print()
cm = ConfusionMatrix(random_forest_credit)
cm.fit(X_credit_treinamento, y_credit_treinamento)
cm.score(X_credit_teste, y_credit_teste)
cm.show()
print(classification_report(y_credit_teste, previsoes))