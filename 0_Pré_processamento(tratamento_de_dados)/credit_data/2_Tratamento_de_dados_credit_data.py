import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

### Importação do banco de dados credit_data ###
credit_data = pd.read_csv('credit_data.csv') # Base de dados histórica
print(credit_data)
# Objetivo dessa base de dados é indentificar o atributo: 'Default'
# atributo resposavél por indicar o clientes que pagam corretamente o banco
# Default == 0 -> Cliente que pagou o emprestimo
# Defaut == 1 -> Cliente que não pagou o emprestimo


### Analizando os dados ###
print(credit_data.describe())
print(np.unique(credit_data['default'], return_counts=True))
# 1717 clientes pagaram e 283 clientes não pagaram
# Indetifica-se uma base de dados 'Desbalanceada'

### Tratamento de valores inconsistentes ###
# foram indentificados algumas idades na coluna 'age' com valores invalidos
# ou negativos.
print(credit_data.loc[credit_data['age'] < 0])
print(credit_data['age'][credit_data['age'] > 0].mean())
# indentificando a médias as idade
media_credit_data = credit_data['age'][credit_data['age'] > 0].mean()

credit_data.loc[credit_data['age'] < 0, 'age'] = media_credit_data
# Feito a subistituição dos valores inconsistente pela média

print(credit_data.head(27))

###Valores vazios###
print(credit_data.isnull().sum())
print(credit_data.loc[pd.isnull(credit_data['age'])])

credit_data['age'].fillna(credit_data['age'].mean(), inplace = True)

print(credit_data.head(32))

### Separar entre previsores e classes ###

X_credit = credit_data.iloc[:, 1:4].values
y_credit = credit_data.iloc[:, 4].values

### Escalonamento dos valores ###

scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)

### Base de treinamento e teste ###
X_credit_treinamento, X_credit_test, y_credit_treinamento, y_credit_teste = (
 train_test_split(X_credit, y_credit, test_size=0.25, random_state=0))

### Salvando-pré-processamento ###

with open('credit.pkl', mode='wb') as f:
    pickle.dump([X_credit_treinamento, y_credit_treinamento,
                 X_credit_test, y_credit_teste], f)

