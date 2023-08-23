import pickle

### Pré_processing ###
with open('credit.pkl', mode='rb') as f:
    X_credit_training, y_credit_training, X_credit_test, y_credit_test =\
        pickle.load(f)

### Training ###
# versoes do kernel svm: 'linear', 'poly', 'rbf', 'sigmoid'

### Prediction ###


### Pós_processing ###