# %%
import numpy as np
import matplotlib.pyplot as plt 
np.random.seed(5)
mean1 = [0, 0]
mean2 = [1, 5]
cov1 = [[1, 0.9], [0.9, 1]]  # diagonal covariance
cov2 = [[1, 0.75], [0.75, 1]]
import matplotlib.pyplot as plt
X1 = np.random.multivariate_normal(mean1, cov1, 5)
X2 = np.random.multivariate_normal(mean2, cov2, 5)
plt.plot(X1[:,0], X1[:,1], 'o')
plt.plot(X2[:,0], X2[:,1], 'o')
#plt.axis('equal')
plt.show()
# %%
X=np.vstack((X1,X2))
# %%
l1 = np.ones((5,1))
l2= -np.ones((5,1))
l=np.vstack((l1,l2))
# %%
dataset = np.hstack((X,l))
# %%
# shuffling the data to make the sampling random
np.random.shuffle(dataset)
m=4
# splitting the data into train/test sets
train = dataset[0:7]
test = dataset[7:10]
