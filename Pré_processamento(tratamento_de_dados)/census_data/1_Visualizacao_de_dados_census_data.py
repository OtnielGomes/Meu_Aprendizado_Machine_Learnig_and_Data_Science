import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

### Pré-precessamento ###
census_data = pd.read_csv('census.csv')
print(census_data)
print(census_data.head())
print(census_data.describe)
print(census_data.isnull().sum())

### Visualização e analise dos dados ###

print(np.unique(census_data['income'], return_counts=True))
# O objetivo nesse banco de dados sera descobrir o salario das
# pessoas, através da coluna 'income' os valores dessa classe
# são definidos em : <=50k ou >50k

# Base de dados com classificação desbalanceada.
plt.hist(x=census_data['age'])
plt.show()
plt.hist(x=census_data['education-num'])
plt.show()
plt.hist(x=census_data['hour-per-week'])
plt.show()
grafico = px.treemap(census_data, path=['workclass', 'age'])
grafico.show()
grafico2 = px.treemap(census_data, path=['occupation', 'relationship', 'age'])
grafico2.show()
grafico3 = px.parallel_categories(census_data,
                                  dimensions=['occupation', 'relationship'])
grafico3.show()
