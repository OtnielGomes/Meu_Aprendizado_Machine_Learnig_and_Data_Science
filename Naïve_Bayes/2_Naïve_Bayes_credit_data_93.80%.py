import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

### Pré_processamento ###
with (open('credit.pkl', mode='rb') as f):
    X_credit_treinamento, y_credit_treinamento,\
    X_credit_teste, y_credit_teste = pickle.load(f)

### Treinamento ###

naive_credit_data = GaussianNB()
naive_credit_data.fit(X_credit_treinamento, y_credit_treinamento)

### Previsoes ###

previsoes = naive_credit_data.predict(X_credit_teste)

### Pós_precessamento ###

precisao_algoritimo = accuracy_score(y_credit_teste,previsoes)
print(precisao_algoritimo)

cm = ConfusionMatrix(naive_credit_data)
cm.fit(X_credit_treinamento, y_credit_treinamento)
cm.score(X_credit_teste, y_credit_teste)
cm.show()
print(classification_report(y_credit_teste, previsoes))