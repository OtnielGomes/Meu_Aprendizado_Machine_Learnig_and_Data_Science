import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

# Loading
house_data = pd.read_csv('house_prices.csv')
X_house = house_data.iloc[:, 3:19].values
y_house = house_data.iloc[:, 2].values

# Split
X_house_training, X_house_test, y_house_training, y_house_test = \
    train_test_split(X_house, y_house, test_size=0.3, random_state=0)

# Salving
with open('house_multi_data.pkl', mode='wb') as f:
    pickle.dump([X_house_training, y_house_training,
                 X_house_test, y_house_test], f)

