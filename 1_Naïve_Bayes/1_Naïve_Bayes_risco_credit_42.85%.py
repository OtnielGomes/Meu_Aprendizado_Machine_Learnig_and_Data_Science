import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

############## Pré processamento #############
risco_credit_data = pd.read_csv('risco_credito.csv')

X_risco_credit = risco_credit_data.iloc[:, 0:4].values
y_risco_credit = risco_credit_data.iloc[:, 4]





label_encoder_historia = LabelEncoder()
label_encoder_divida = LabelEncoder()
label_encoder_garantia = LabelEncoder()
label_encoder_renda = LabelEncoder()

X_risco_credit[:,0] = (label_encoder_historia.fit_transform
                        (X_risco_credit[:, 0]))
X_risco_credit[:,1] = (label_encoder_divida.fit_transform
                        (X_risco_credit[:, 1]))
X_risco_credit[:,2] = (label_encoder_garantia.fit_transform
                        (X_risco_credit[:, 2]))
X_risco_credit[:,3] = (label_encoder_renda.fit_transform
                        (X_risco_credit[:, 3]))

################## Treinamento ################
naive_risco_credito = GaussianNB()
naive_risco_credito.fit(X_risco_credit, y_risco_credit)

################# Previsão ####################

# historia Boa (0), divida Alta (0), garantia Nenhuma (1), renda >35 (2)
# historia Ruim (2), divida Alta (0), garantia Adequada (0), renda < 15 (0)
previsao = naive_risco_credito.predict([[0, 0, 1, 2], [2, 0, 0, 0]])
print(previsao)
print(naive_risco_credito.classes_)
print(naive_risco_credito.class_count_)
print(naive_risco_credito.class_prior_)
