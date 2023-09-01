import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle
### Importação do banco de dados census_data ###
census_data = pd.read_csv('census.csv')
print(census_data)
print(census_data.head())
print(census_data.describe)
print(census_data.isnull().sum())
print(census_data.columns)

# Todos os dados da base de dados estão completos e preenchidos
# portanto não há necessidade de tratar dados inconsistentes ou nulos

### Separação entre previsores e classes ###

X_census = census_data.iloc[:, 0:14].values
y_census = census_data.iloc[:, 14].values

### Tratando atributos categóricos ###

#tratando com label encoder
label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

X_census[:, 1] = label_encoder_workclass.fit_transform(X_census[:, 1])
X_census[:, 3] = label_encoder_education.fit_transform(X_census[:, 3])
X_census[:, 5] = label_encoder_marital.fit_transform(X_census[:, 5])
X_census[:, 6] = label_encoder_occupation.fit_transform(X_census[:, 6])
X_census[:, 7] = label_encoder_relationship.fit_transform(X_census[:, 7])
X_census[:, 8] = label_encoder_race.fit_transform(X_census[:, 8])
X_census[:, 9] = label_encoder_sex.fit_transform(X_census[:, 9])
X_census[:, 13] = label_encoder_country.fit_transform(X_census[:, 13])

#ultilizando o onehotencoder para balancear os atributos categóricos

onehotencoder_census = ColumnTransformer(transformers=[('OneHot',
        OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')
X_census = onehotencoder_census.fit_transform(X_census).toarray()
print(X_census.shape)

### Ecalonamento dos atributos ###

scaler_census = StandardScaler()
X_census = scaler_census.fit_transform(X_census)

### Base de treinamento e teste ###

X_census_training, X_census_test, y_census_training, y_census_test = (
train_test_split(X_census, y_census, test_size=0.15, random_state=0))

### Salvando-pré-processamento ###

with open('census_data.pkl', mode='wb') as f:
    pickle.dump([X_census_training, y_census_training,
                 X_census_test, y_census_test], f)

