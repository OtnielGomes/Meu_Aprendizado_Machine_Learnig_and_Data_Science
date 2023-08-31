from sklearn.model_selection import cross_val_score, KFold
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import pickle

with open('credit_cross.pkl', mode='rb') as f:
    X_credit, y_credit = pickle.load(f)
help(MLPClassifier)
result_neural_network \
    = list()
for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)

    neural_network = MLPClassifier(activation='relu',
                                   batch_size=56,
                                   hidden_layer_sizes=10,
                                   max_iter=30000,
                                   solver='adam',
                                   tol=1e-08,
                                   verbose=True)
    score_neural_network = cross_val_score(neural_network,
                                           X_credit, y_credit,
                                           cv=kfold)
    result_neural_network.append(score_neural_network.mean())

results = {'Neural Network': result_neural_network}
df_result = pd.DataFrame(results)
df_result.to_csv('Data_results1.csv')

algorithms_scores = {'Accuracy': np.concatenate([result_neural_network]),
                     'Algorithms':  ['Neural'] * len(result_neural_network)}
df_algorithms_scores = pd.DataFrame(algorithms_scores)
df_algorithms_scores.to_csv('algorithms_scores1.csv')
