import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#----Compulsory Task1----

f = pd.read_csv("data2008.csv")
x = f.iloc[:,1:2].values
y = f.iloc[:,2:3].values

# Find the Euclidean distance between two points
def distance (x1, y1, x2, y2):
    dist = ((x1-x2)**2 + (y1-y2)**2)**.5
    return(dist)

print("Number of Countries: ", f['Countries'].count())
print("Countries: ", f["Countries"].unique())
print("Mean for Birthrate: ", f['BirthRate(Per1000 - 2008)'].mean())
print('Mean for LifeExpectancy: ', f['LifeExpectancy(2008)'].mean())

print(distance(1, 2, 100, 150))

# Find Centroid center coordinates

coordinatescenter = []

def findCenter(xList, yList, coordinates):
    xCenter = np.sum(xList)/len(xList)
    yCenter = np.sum(yList)/len(yList)
    coordinates.append(xCenter)
    coordinates.append(yCenter)
    return coordinates

findCenter(x, y, coordinatescenter)

# List to Numpy array
np.array(coordinatescenter)

plt.scatter(x, y, label = "Point (x;y)", color = "k")
plt.scatter(coordinatescenter[0], coordinatescenter[1], label = 'centroid(x;y)', color = 'k')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

def kmeans(X, k, maxiter, seed=None):
    """
    specify the number of clusters k and
    the maximum iteration to run the algorithm
    """
    n_row, n_col = X.shape

    # Randomly choose k data points as initial centroids
    if seed is not None:
        np.random.seed(seed)

    rand_indices = np.random.choice(n_row, size=k)
    centroids = X[rand_indices]

    for itr in range(maxiter):
        # Compute distances between each data point and the set of centroids
        # and assign each data point to the closest centroid
        distances_to_centroids = pairwise_distances(X, centroids, metric='euclidean')
        cluster_assignment = np.argmin(distances_to_centroids, axis=1)

        # Select all data points that belong to cluster i and compute
        # the mean of these data points (each feature individually)
        # this will be our new cluster centroids
        new_centroids = np.array([X[cluster_assignment == i].mean(axis=0) for i in range(k)])

        # If the updated centroid is still the same,
        # then the algorithm converged
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    heterogeneity = 0
    for i in range(k):
        # note that pairwise_distance only accepts 2d-array
        cluster_data = X[cluster_assignment == i]
        distances = pairwise_distances(cluster_data, [centroids[i]], metric='euclidean')
        heterogeneity += np.sum(distances ** 2)

    return centroids, cluster_assignment, heterogeneity

# Compare difference between data1953.csv and data2008.csv
f2 = pd.read_csv('data1953.csv')
c = pd.concat([f,f2], axis=0)

c.drop_duplicates(keep='first', inplace=True)
c.reset_index(drop=True, inplace=True)
print(c)