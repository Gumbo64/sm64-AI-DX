import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
# Define the vector and the list of vectors
dimensionality = 10
data = np.random.rand(100, dimensionality)
# Define a query vector
query_vector = np.random.rand(dimensionality)

# Calculate Euclidean distances between the query vector and the dataset
distances = euclidean_distances(data, [query_vector])

# Find the closest vector
closest_index = np.argmin(distances)
closest_vector = data[closest_index]

print(f"Closest vector: {closest_vector}")
