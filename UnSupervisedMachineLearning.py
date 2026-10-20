from sklearn.cluster import KMeans
# Customers: [age, spending score], no labels given
X = [
    [20, 80], [22, 85], [25, 90],      # young, high spenders
    [50, 20], [55, 15], [60, 25],      # older, low spenders
]

model = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X) 
#n_init represents the number of time the code runs with different centre values and 
# returns the best value of the centre. It is used to avoid local minima.

print(model.labels_)  # Cluster labels for each customer
print(model.cluster_centers_)  # Coordinates of cluster centers
print(model.predict([[23,88]]))  # Predicting the cluster for a new customer

# -----------------------------------------------------------------------------------------
# UNSUPERVISED MACHINE LEARNING for 3D point cloud data
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Synthetic point cloud: 3 blobs in 3D, no labels used
rng = np.random.default_rng(0)
X = np.vstack([
    rng.normal([0, 0, 0], 0.3, (100, 3)),   # object 1 [centered at (0,0,0), standard deviation 0.3, (100 points,each point is 3Dimensional)]
    rng.normal([3, 3, 0], 0.3, (100, 3)),   # object 2
    rng.normal([0, 3, 3], 0.3, (100, 3)),   # object 3
])                                          # shape: (300, 3)

model = KMeans(n_clusters=6, random_state=0, n_init=10).fit(X)

print(X.shape)
print(model.labels_[:100]) # Cluster labels for the first 10 points
print(model.cluster_centers_)  # Coordinates of cluster centers

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(X[:,0],X[:,1],X[:,2] ,c=model.labels_,s =10)
plt.show()

# -----------------------------------------------------------------------------------------