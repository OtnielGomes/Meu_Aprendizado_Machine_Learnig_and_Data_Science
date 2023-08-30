from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import pickle
# Pre-processing
with open('credit_cross.pkl', mode='rb') as f:
    X_credit, y_credit = pickle.load(f)

result_decision_tree = list()
result_random_forest = list()
result_knn = list()
result_logist_regression = list()
result_svm = list()
result_neural_network = list()

for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)

    # Decision Tree
    decision_tree = DecisionTreeClassifier(criterion='entropy',
                                           min_samples_leaf=1,
                                           min_samples_split=5,
                                           splitter='best')
    score_decision_tree = cross_val_score(decision_tree,
                                          X_credit, y_credit,
                                          cv=kfold)
    result_decision_tree.append(score_decision_tree.mean())

    # Random Forest
    random_forest = RandomForestClassifier(criterion='entropy',
                                           min_samples_leaf=1,
                                           min_samples_split=2,
                                           n_estimators=100)
    score_random_forest = cross_val_score(random_forest,
                                          X_credit, y_credit,
                                          cv=kfold)
    result_random_forest.append(score_random_forest.mean())

    # KNN
    knn = KNeighborsClassifier(n_neighbors=20,
                               p=1,
                               weights='distance')
    score_knn = cross_val_score(knn,
                                X_credit, y_credit,
                                cv=kfold)
    result_knn.append(score_knn.mean())

    # Logistic Regression
    logistic_regression = LogisticRegression(C=1.0,
                                             solver='lbfgs',
                                             tol=0.0001)
    score_logistic_regression = cross_val_score(logistic_regression,
                                                X_credit, y_credit,
                                                cv=kfold)
    result_logist_regression.append(score_logistic_regression.mean())

    # SVM
    svm = SVC(C=1.5,
              kernel='rbf',
              tol=0.001)
    score_svm = cross_val_score(svm,
                                X_credit, y_credit,
                                cv=kfold)
    result_svm.append(score_svm.mean())

    # Neural_Network
    neural_network = MLPClassifier(activation='relu',
                                   batch_size=1024,
                                   hidden_layer_sizes=(2, 2),
                                   max_iter=30000,
                                   solver='adam',
                                   tol=1e-08)
    score_neural_network = cross_val_score(neural_network,
                                           X_credit, y_credit,
                                           cv=kfold)
    result_neural_network.append(score_neural_network.mean())


# Pos-processing/ salve data

results = {'Decision Tree': result_decision_tree,
           'Random Forest': result_random_forest,
           'KNN': result_knn,
           'Logistic Regression': result_logist_regression,
           'SVM': result_svm,
           'Neural Network': result_neural_network}
df_result = pd.DataFrame(results)
df_result.to_cvs('Data_results')

algorithms_scores = {'Accuracy': np.concatenate([result_decision_tree,
                                                 result_random_forest,
                                                 result_knn,
                                                 result_logist_regression,
                                                 result_svm,
                                                 result_neural_network]),
                     'Algorithms': ['Tree', 'Tree', 'Tree', 'Tree', 'Tree',
                                    'Tree', 'Tree', 'Tree', 'Tree', 'Tree',
                                    'Tree', 'Tree', 'Tree', 'Tree', 'Tree',
                                    'Tree', 'Tree', 'Tree', 'Tree', 'Tree',
                                    'Tree', 'Tree', 'Tree', 'Tree', 'Tree',
                                    'Tree', 'Tree', 'Tree', 'Tree', 'Tree',
                                    'Random_Forest', 'Random_forest',
                                    'Random_forest', 'Random_forest',
                                    'Random_forest', 'Random_forest',
                                    'Random_forest', 'Random_forest',
                                    'Random_forest', 'Random_forest',
                                    'Random_forest', 'Random_forest',
                                    'Random_forest', 'Random_forest',
                                    'Random_forest', 'Random_forest',
                                    'Random_forest', 'Random_forest',
                                    'Random_forest', 'Random_forest',
                                    'Random_forest', 'Random_forest',
                                    'Random_forest', 'Random_forest',
                                    'Random_forest', 'Random_forest',
                                    'Random_forest', 'Random_forest',
                                    'Random_forest', 'Random_forest',
                                    'KNN', 'KNN', 'KNN', 'KNN', 'KNN', 'KNN',
                                    'KNN', 'KNN', 'KNN', 'KNN', 'KNN', 'KNN',
                                    'KNN', 'KNN', 'KNN', 'KNN', 'KNN', 'KNN',
                                    'KNN', 'KNN', 'KNN', 'KNN', 'KNN', 'KNN',
                                    'KNN', 'KNN', 'KNN', 'KNN', 'KNN', 'KNN',
                                    'Logistic', 'Logistic', 'Logistic',
                                    'Logistic', 'Logistic', 'Logistic',
                                    'Logistic', 'Logistic', 'Logistic',
                                    'Logistic', 'Logistic', 'Logistic',
                                    'Logistic', 'Logistic', 'Logistic',
                                    'Logistic', 'Logistic', 'Logistic',
                                    'Logistic', 'Logistic', 'Logistic',
                                    'Logistic', 'Logistic', 'Logistic',
                                    'Logistic', 'Logistic', 'Logistic',
                                    'Logistic', 'Logistic', 'Logistic',
                                    'SVM', 'SVM', 'SVM', 'SVM', 'SVM', 'SVM',
                                    'SVM', 'SVM', 'SVM', 'SVM', 'SVM', 'SVM',
                                    'SVM', 'SVM', 'SVM', 'SVM', 'SVM', 'SVM',
                                    'SVM', 'SVM', 'SVM', 'SVM', 'SVM', 'SVM',
                                    'SVM', 'SVM', 'SVM', 'SVM', 'SVM', 'SVM',
                                    'Neural', 'Neural', 'Neural', 'Neural',
                                    'Neural', 'Neural', 'Neural', 'Neural',
                                    'Neural', 'Neural', 'Neural', 'Neural',
                                    'Neural', 'Neural', 'Neural', 'Neural',
                                    'Neural', 'Neural', 'Neural', 'Neural',
                                    'Neural', 'Neural', 'Neural', 'Neural',
                                    'Neural', 'Neural', 'Neural', 'Neural',
                                    'Neural', 'Neural']}
df_algorithms_scores = pd.DataFrame(algorithms_scores)
df_algorithms_scores.to_cvs('algorithms_scores')

