import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

### Pré-precessamento ###
base_risco_credito = pd.read_csv('risco_credito.csv')

### Separação entre previsores e classes ###
X_risco_credito = base_risco_credito.iloc[:, 0:4].values
y_risco_credito = base_risco_credito.iloc[:, 4].values

### Tratando atributos categóricos ###
label_encoder_historia = LabelEncoder()
label_encoder_divida = LabelEncoder()
label_encoder_garantia = LabelEncoder()
label_encoder_renda = LabelEncoder()

X_risco_credito[:, 0] = (label_encoder_historia.fit_transform(
                                                        X_risco_credito[:, 0]))
X_risco_credito[:, 1] = (label_encoder_divida.fit_transform(
                                                        X_risco_credito[:, 1]))
X_risco_credito[:, 2] = (label_encoder_garantia.fit_transform(
                                                        X_risco_credito[:, 2]))
X_risco_credito[:, 3] = (label_encoder_renda.fit_transform(
                                                        X_risco_credito[:, 3]))
print(X_risco_credito)

with open('risco_credito.pkl', 'wb') as f:
    pickle.dump([X_risco_credito, y_risco_credito], f)

