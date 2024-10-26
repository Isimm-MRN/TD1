# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def load_data():
    URL_='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    data = pd.read_csv(URL_, header = None)   
    # Rendre l'ensemble de jeux de données separable
    data = data[:100]
    data[4] = np.where(data.iloc[:, -1]=='Iris-setosa', -1, 1)
    data = np.asmatrix(data, dtype = 'float64')
    return data
data = load_data()
X = data[:, :-1]
L = data[:, -1]

# %%
plt.scatter(np.array(X[:50,0]), np.array(X[:50,2]), marker='o', label='setosa')
plt.scatter(np.array(X[50:,0]), np.array(X[50:,2]), marker='o', label='versicolor')
plt.xlabel('Longueur du pétale')
plt.ylabel('Longueur du sépale')
plt.legend()
plt.show()


