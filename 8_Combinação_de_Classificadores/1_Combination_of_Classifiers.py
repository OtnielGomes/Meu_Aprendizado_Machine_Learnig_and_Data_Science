import pickle


# Pre-processing
with open('credit_cross.pkl', mode='rb') as f:
    X_credit, y_credit = pickle.load(f)

# Loading classifiers

decision_tree = pickle.load(open('tree_classifier.sav', mode='rb'))
svm = pickle.load(open('svm_classifier.sav', mode='rb'))
neural_network = pickle.load(open('neural_classifier.sav', mode='rb'))

# Loading register for testing
new_register = X_credit[1999]
new_register_reshape = new_register.reshape(1, -1)

# Prediction
prediction_decision_tree = decision_tree.predict(new_register_reshape)
prediction_svm = svm.predict(new_register_reshape)
prediction_neural_network = neural_network.predict(new_register_reshape)

# Pos-processing

# Individual decision
# As respostas são 0>>> para os cliente que pagam e 1>>> para o clientes que
# não pagam
if prediction_decision_tree[0] == 0:
    print('Previsão da Arvore de Decisão: O cliente vai pagar o empréstimo')
else:
    print('Previsão da Arvore de Decisão: O cliente não pagará o emprestimo')
if prediction_svm[0] == 0:
    print('Previsão do SVM : O cliente vai pagar o empréstimo')
else:
    print('Previsão do SVM: O cliente não pagará o emprestimo')
if prediction_decision_tree[0] == 0:
    print('Previsão da Rede Neural: O cliente vai pagar o empréstimo')
else:
    print('Previsão da Rede Neural: O cliente não pagará o emprestimo')

# Combination classifiers

print('\nResposta conjunta dos Algoritmos')
pay = 0
not_pay = 0

if prediction_decision_tree[0] == 0:
    pay += 1
else:
    not_pay +=1
if prediction_svm[0] == 0:
    pay += 1
else:
    not_pay += 1
if prediction_neural_network[0] == 0:
    pay += 1
else:
    not_pay += 1
if pay > not_pay:
    print('De acordo com os algoritmos o cliente Pagará o empréstimo')
elif pay == not_pay:
    print('Não há uma resposta acertiva dos algoritmos')
else:
    print('De acordo com os algoritmos o cliente Não pagará o emprestimo')

