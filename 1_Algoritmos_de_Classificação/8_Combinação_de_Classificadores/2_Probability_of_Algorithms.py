import pickle


# Pre-processing
with open('credit_cross.pkl', mode='rb') as f:
    X_credit, y_credit = pickle.load(f)

# Loading classifiers

decision_tree = pickle.load(open('tree_classifier.sav', mode='rb'))
svm = pickle.load(open('svm_classifier.sav', mode='rb'))
neural_network = pickle.load(open('neural_classifier.sav', mode='rb'))

# Loading register for testing
new_register = X_credit[5]
new_register_reshape = new_register.reshape(1, -1)

# Prediction
prediction_decision_tree = decision_tree.predict(new_register_reshape)
prediction_svm = svm.predict(new_register_reshape)
prediction_neural_network = neural_network.predict(new_register_reshape)

# Probability algorithms

# Decision Tree
probability_decision_tree = decision_tree.predict_proba(new_register_reshape)
reliability_decision_tree = probability_decision_tree.max()

# SVM
probability_svm = svm.predict_proba(new_register_reshape)
reliability_svm = probability_svm.max()

# Neural Network
probability_neural_network = neural_network.predict_proba(new_register_reshape)
reliability_neural_network = probability_neural_network.max()

# Pos-processing
print(f'Confiança do Algoritmo Decision Tree: {reliability_decision_tree}')
print(f'Confiança do Algoritmo SVM : {reliability_svm}')
print(f'Confiaça do Algoritmo de Redes Neurais : {reliability_neural_network}')

# Combination classifiers + Reliability Min

reliability_min = 0.99999
algorithms_used = 0
pay = 0
not_pay = 0

if reliability_decision_tree >= reliability_min:
    algorithms_used += 1
    if prediction_decision_tree[0] == 0:
        pay += 1
    else:
        not_pay += 1
if reliability_svm >= reliability_min:
    algorithms_used += 1
    if prediction_svm[0] == 0:
        pay += 1
    else:
        not_pay += 1
if reliability_neural_network >= reliability_min:
    algorithms_used += 1
    if prediction_neural_network[0] == 0:
        pay += 1
    else:
        not_pay += 1

if pay > not_pay:
    print(f'De acordo com {algorithms_used} Algoritmos. O cliente Pagará o '
          f'empréstimo')
elif pay == not_pay:
    print('Não há uma resposta acertiva dos algoritmos')
else:
    print(f'De acordo com {algorithms_used} Algoritmos. O cliente Não pagará o '
          f'emprestimo')