import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

### Pré_processing ###
with open('census_data.pkl', mode='rb') as f:
    X_census_training, y_census_training, X_census_test, y_census_test =\
            pickle.load(f)

### Training ###
#solver's: 'lbfgs', 'sgd', 'adam'
#activation: 'relu', 'logistic', 'tanh'
#Durante o processo de aprendizagem foram feito alguns testes e foi
# identificado que os melhores parametros foram:
#Activation:relu
#Solver:adam
#camadas ocultas: 2 camadas com 55 neurônio cada usando a formula:
# (classificadores + saida) / 2 >>> 108+1= 109/2 = 54,5 >>> 55
neural_network_credit = MLPClassifier(max_iter=1000,
                                      verbose=True,
                                      tol=0.00000100,
                                      solver='adam',
                                      activation='relu',
                                      hidden_layer_sizes=(55, 55))
neural_network_credit.fit(X_census_training, y_census_training)

### Prediction ###
prediction = neural_network_credit.predict(X_census_test)
prediction_accuracy = accuracy_score(y_census_test, prediction)


### Pós_processing ###
def main():
    print(prediction_accuracy)
    cm = ConfusionMatrix(neural_network_credit)
    cm.fit(X_census_training, y_census_training)
    cm.score(X_census_test, y_census_test)
    cm.show()
    print(classification_report(y_census_test, prediction))


if __name__ == '__main__':
    main()

