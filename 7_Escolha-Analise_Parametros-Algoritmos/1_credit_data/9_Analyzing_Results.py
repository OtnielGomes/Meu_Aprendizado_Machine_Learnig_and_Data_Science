import pandas as pd
from scipy.stats import shapiro, f_oneway
from statsmodels.stats.multicomp import MultiComparison
import seaborn as sns
import matplotlib.pyplot as plt

# Pre-processing
pd.options.display.max_rows = None
pd.options.display.max_columns = None

results = pd.read_csv('Data_results.csv')
print(results.describe())

# Comparação de Algoritmos
# Definimos o valor padão de 0.05
# Alpha
# Probabilidade de rejeitar a
# Hípotese Nula, quanto menor mais
# seguro o resultado
alpha = 0.05

comparison_algorithms = shapiro(results['Decision Tree']), \
                        shapiro(results['Random Forest']), \
                        shapiro(results['KNN']), \
                        shapiro(results['Logistic Regression']), \
                        shapiro(results['SVM']), \
                        shapiro(results['Neural Network'])

print(comparison_algorithms)
# Graphics

sns.displot(data=results, x='Decision Tree', bins=20, kde=True )
plt.show()
sns.displot(data=results, x='Random Forest', bins=20, kde=True )
plt.show()
sns.displot(data=results, x='KNN', bins=20, kde=True )
plt.show()
sns.displot(data=results, x='Logistic Regression', bins=20, kde=True )
plt.show()
sns.displot(data=results, x='SVM', bins=20, kde=True )
plt.show()
sns.displot(data=results, x='Neural Network', bins=20, kde=True )
plt.show()

# Hypothesis test ANOVA e Tukey
_, p = f_oneway(results['Decision Tree'],
                results['Random Forest'],
                results['KNN'],
                results['Logistic Regression'],
                results['SVM'],
                results['Neural Network'])
print(p)
if p <= alpha:
    print('Hipótese nula rejeitada. Os dados são diferentes')
else:
    print('Hipótese alternativa rejeitada. Os resultados são os mesmos\n')

# Testing Tukey
algorithms_scores = pd.read_csv('algorithms_scores.csv')
print(algorithms_scores)

comparing_algorithms = MultiComparison(algorithms_scores['Accuracy'],
                                       algorithms_scores['Algorithms'])
statistical_test = comparing_algorithms.tukeyhsd()
print(statistical_test)
print(results.mean())
statistical_test.plot_simultaneous()
plt.show()

# Considerando que a hipótese nula não foi aceita, entende-se
# que os algoritmos são estatisticamente diferentes então Rede_Neural tem o
# maior precisão dentre elas, entende-se que a Rede_Neural é
#o algoritmo com a maior precisão.
