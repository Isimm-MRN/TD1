# %%
import numpy as np
from matplotlib import pyplot as plt
# %% 
# # Algorithme d'apprentissage d'un perceptron.
# %%
def perceptron(X, L):
    w = np.zeros(len(X[0])+1)    
    kmax = 20
    for t in range(kmax):
        for i, x in enumerate(X):
            Xi=np.append(X[i],np.ones(1))
            if (np.dot(w,Xi)*L[i]) <= 0:
                w = w + Xi*L[i]
    return w
# %% 
# # Cas d'un ensemble de jeux des données simple.
# %%
X = np.array([[-2,4],[4,1],[1, 6],[2, 4],[6, 2],])
L = np.array([-1,-1,1,1,1])
# Visualiser une droite qui separe les jeux de données.
plt.plot([-2,6],[6,0.5],"--")
plt.legend(['La droite séparante des données'], loc="lower right") 
for d, sample in enumerate(X):
    # Visualiser les données negatives.
    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2,c="blue")
    # Visualiser les données positives.
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2,c="red")
plt.show()
# %% 
# ## Visualiser la droite donnée par ce perceptron

# %%
W = perceptron(X,L)
print("Le poids de ce perceptron est W=",W)
# Visualiser la droite donnée par ce perceptron.
x = np.linspace(-7,7,100)
plt.plot(x,  -W[2]/W[1]-W[0]*x/W[1] , c="red")
plt.legend(['La droite donnée par le perceptron'], loc="lower right")
for d, sample in enumerate(X):
    # Visualiser les données negatives.
    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2,color="blue")
    # Visualiser les données positves.
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2,color="red")
plt.show()
# %% 
# ## Visualiser les données tests.
# %%
# Visualiser les données tests.
# Visualiser la droite donnée par ce perceptron.
x = np.linspace(-7,7,100)
plt.plot(x,  -W[2]/W[1]-W[0]*x/W[1] , c="red")
plt.legend(['Visualiser les données tests']) 
plt.scatter(2,2, s=120, marker='_', linewidths=2, color='black')
plt.scatter(4,3, s=120, marker='+', linewidths=2, color='black')
plt.show()
# %% 
# # Cas d'un ensemble de jeux des données aléatoires.
# %%
mean1 = [0, 0]
mean2 = [1, 5]
cov1 = [[1, 0.9], [0.9, 1]]   
cov2 = [[1, 0.75], [0.75, 1]]
np.random.seed(5)
X1 = np.random.multivariate_normal(mean1, cov1, 5)
X2 = np.random.multivariate_normal(mean2, cov2, 5)
# %%
X=np.vstack((X1,X2))
l1 = np.ones((5,1))
l2= -np.ones((5,1))
L=np.vstack((l1,l2))
# Visualiser une droite qui separe les jeux de données.
plt.plot([-4,3],[6,2],"--", c="blue")
plt.legend(['La droite séparante des données'], loc="lower right") 
plt.plot(X1[:,0], X1[:,1], 'o')
plt.plot(X2[:,0], X2[:,1], 'o')
plt.show()
# %%
W = perceptron(X,L)
print("Le poids de ce perceptron est W=",W)
# Visualiser la droite   donnée par le perceptron.
x = np.linspace(-7,7,100)
plt.plot(x,  -W[2]/W[1]-W[0]*x/W[1] , c="red")
plt.axis('equal')
plt.legend(['La droite donnée par le perceptron'], loc="lower right") 
plt.plot(X1[:,0], X1[:,1], 'o')
plt.plot(X2[:,0], X2[:,1], 'o')
plt.show()
