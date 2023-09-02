import pandas as pd

# Loading
pd.options.display.max_columns = None
# Função ultilizada para ver todas as colunas no pycharm
house_data = pd.read_csv('house_prices.csv')
print(house_data)
print(house_data.describe())
print(house_data.isnull().sum())
# Verificado que todas as colunas estão preenchidas corretamente
# Apenas a coluna de datas que possui algumas strings e impede que verifiquemos
# A Correlação dos dados
house_data_corr = house_data.drop(['date'], axis=1)

print(house_data_corr.corr())
