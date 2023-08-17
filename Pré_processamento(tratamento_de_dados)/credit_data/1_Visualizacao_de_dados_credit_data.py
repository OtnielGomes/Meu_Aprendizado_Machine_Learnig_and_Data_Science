import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


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

plt.hist(x=(credit_data['age']))
plt.show()
plt.hist(x=(credit_data['loan']))
plt.show()
plt.hist(x=(credit_data['income']))
plt.show()
grafico = px.scatter_matrix(credit_data, dimensions=['age', 'income', 'loan'], color='default')
grafico.show()


