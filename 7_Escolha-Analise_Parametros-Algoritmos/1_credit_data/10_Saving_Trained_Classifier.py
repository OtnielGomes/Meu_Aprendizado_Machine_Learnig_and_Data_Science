import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Pre-Processing
with open('credit_cross.pkl', mode='rb') as f:
    X_credit, y_credit = pickle.load(f)

# Training

# Decision Tree
tree_classifier = DecisionTreeClassifier(criterion='entropy',
                                         min_samples_leaf=1,
                                         min_samples_split=5,
                                         splitter='best')
tree_classifier.fit(X_credit, y_credit)

# SVM
svm_classifier = SVC(C=2.0,
                     kernel='rbf',
                     tol=0.001)
svm_classifier.fit(X_credit, y_credit)

# Neural Network
neural_classifier = MLPClassifier(activation='relu',
                                  batch_size=56,
                                  hidden_layer_sizes=10,
                                  max_iter=30000,
                                  solver='adam',
                                  tol=1e-08)
neural_classifier.fit(X_credit, y_credit)

# Saving
pickle.dump(tree_classifier, open('tree_classifier.sav', mode='wb'))
pickle.dump(svm_classifier, open('svm_classifier.sav', mode='wb'))
pickle.dump(neural_classifier, open('neural_classifier.sav', mode='wb'))
