import torch
import cudf
import timeit
from cuml import NearestNeighbors

# Create a random point cloud
n_points = 65536
n_neighbors = 16

# Generate a "coordinate" dataframe
df = cudf.DataFrame()
df['x'] = torch.cuda.FloatTensor(n_points).uniform_()
df['y'] = torch.cuda.FloatTensor(n_points).uniform_()
df['z'] = torch.cuda.FloatTensor(n_points).uniform_()


# Create a cuML NearestNeighbors model
nn = NearestNeighbors(n_neighbors=n_neighbors)


start_time = timeit.default_timer()

# Fit the model with the input data
nn.fit(df)

# Get the nearest neighbors
distances, indices = nn.kneighbors(df)

end_time = timeit.default_timer()
execution_time = end_time - start_time
print(f"auto Execution time: {execution_time} seconds")


# Create a cuML NearestNeighbors model
nn2 = NearestNeighbors(n_neighbors=n_neighbors, algorithm='rbc')


start_time = timeit.default_timer()

# Fit the model with the input data
nn2.fit(df)

# Get the nearest neighbors
distances, indices = nn2.kneighbors(df)

end_time = timeit.default_timer()
execution_time = end_time - start_time
print(f"rbc Execution time: {execution_time} seconds")


# Print the results
print("Distances:\n", distances)
print("Indices:\n", indices)

print("Aqui se acaba el ejemplo")
