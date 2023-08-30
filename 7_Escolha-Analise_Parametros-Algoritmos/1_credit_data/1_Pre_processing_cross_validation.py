import numpy as np
import pickle

# Para fazer a análise e testes estarei fazendo a junção das partes de
# treino e test: X_training + X_test e y_training + y_teste para implementação
# com os testes através do cross-validation

with open('credit.pkl', mode='rb') as f:
    X_credit_training, y_credit_training, X_credit_test, y_credit_test = \
        pickle.load(f)
X_credit = np.concatenate((X_credit_training, X_credit_test), axis=0)
y_credit = np.concatenate((y_credit_training, y_credit_test), axis=0)

with open('credit_cross.pkl', mode='wb') as f:
    pickle.dump([X_credit, y_credit], f)
