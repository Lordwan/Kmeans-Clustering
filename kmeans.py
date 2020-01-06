import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math


def euclidean_distance(p0, p1):
    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


df = pd.read_csv('data1953.csv')
feat_cols = ["BirthRate(Per1000 - 1953)", "LifeExpectancy(1953)"]
features = np.array(df[feat_cols])

for i in features:
    my_data = []
    new_data = euclidean_distance(features[0], features[1])
    my_data.append(new_data)

k = 6
max_iterations = 300
kmeans = KMeans(n_clusters=k, max_iter=max_iterations)
kmeans.fit(features)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

fig = plt.figure()
colors = ['r', 'b', 'g', 'y', 'c', 'm']

countries = df.iloc[:, 0]

for i in range(len(features)):
    plt.scatter(features[i][0], features[i][1], c=colors[labels[i]])
    plt.scatter(centroids[:, 0], centroids[:, 1], c="black")

clusters = {}

for i in range(k):
    clusters[i] = []

for i in range(len(labels)):
    clusters[labels[i]].append(countries[i])

print("Total countries belonging to cluster 1 is: " + str(len(clusters[0])))
print("Total countries belonging to cluster 2 is: " + str(len(clusters[1])))
print("Total countries belonging to cluster 3 is: " + str(len(clusters[2])))
print("Total countries belonging to cluster 4 is: " + str(len(clusters[3])))
print("Total countries belonging to cluster 5 is: " + str(len(clusters[4])))
print("Total countries belonging to cluster 6 is: " + str(len(clusters[5])))
print('\n')

print("Countries in cluster 1: " + str(clusters[0]) + '\n')
print("Countries in cluster 2: " + str(clusters[1]) + '\n')
print("Countries in cluster 3: " + str(clusters[2]) + '\n')
print("Countries in cluster 4: " + str(clusters[3]) + '\n')
print("Countries in cluster 5: " + str(clusters[4]) + '\n')
print("Countries in cluster 6: " + str(clusters[5]) + '\n')

print("\n" + "Mean Life Expectancy / Birth Rate per cluster")
print(centroids)

for i in range(len(labels)):
    my_data = euclidean_distance(features[:, 0], features[:, 1])

plt.xlabel('BirthRate(Per 1000)')
plt.ylabel('Life Expectancy(Years)')
plt.title('Countries of the world birthrate vs life expectancy')
plt.show()
