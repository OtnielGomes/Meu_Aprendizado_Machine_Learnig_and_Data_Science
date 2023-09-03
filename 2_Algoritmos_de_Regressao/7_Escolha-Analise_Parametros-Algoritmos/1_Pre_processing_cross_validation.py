import pickle
import numpy as np

# Para fazer a análise e testes estarei fazendo a junção das partes de
# treino e test: X_training + X_test e y_training + y_teste para implementação
# com os testes através do cross-validation

with open('house_multi_data.pkl', mode='rb') as f:
    X_house_training, y_house_training, X_house_test, y_house_test = \
        pickle.load(f)
X_house = np.concatenate((X_house_training, X_house_test), axis=0)
y_house = np.concatenate((y_house_training, y_house_test), axis=0)

with open('house_cross_data.pkl', mode='wb') as f:
    pickle.dump([X_house, y_house], f)

