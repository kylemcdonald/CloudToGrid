import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from time import time
import argparse
from scipy.optimize import linear_sum_assignment
import lapcsa
import umap
from auction_lap import auction_lap
import json
import os

# Set random seed
np.random.seed(0)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("nx", type=int, help="Number of grid points in x direction")
parser.add_argument("ny", type=int, help="Number of grid points in y direction")
parser.add_argument("input_json", type=str, help="Path to input JSON file containing high-dimensional data")
parser.add_argument("--solver", type=str, choices=['scipy', 'lapcsa', 'auction'], default='scipy',
                   help="Assignment solver to use: 'scipy', 'lapcsa', or 'auction'")
parser.add_argument("--auction_eps", type=float, default=0.01,
                   help="Epsilon parameter for auction algorithm")
parser.add_argument("--min_epsilon_factor", type=float, default=2.0,
                   help="Minimum epsilon factor for lapcsa algorithm")
parser.add_argument("--scale_factor", type=float, default=10.0,
                   help="Scale factor for lapcsa algorithm")
args = parser.parse_args()

# Calculate total number of points needed
n_points = args.nx * args.ny

# Load high-dimensional data from JSON
print("Loading data from JSON...")
with open(args.input_json, 'r') as f:
    data = np.array(json.load(f))
if len(data) < n_points:
    raise ValueError(f"Input data has {len(data)} points, but {n_points} points are needed for {args.nx}x{args.ny} grid")
data = data[:n_points]  # Take only the number of points we need

# Check if cached transformation exists
cache_file = 'fit_transform.json'
if os.path.exists(cache_file):
    print("Loading cached UMAP transformation...")
    with open(cache_file, 'r') as f:
        point_cloud = np.array(json.load(f))
    point_cloud = point_cloud[:n_points]  # Take only the number of points we need
else:
    # Reduce to 2D using UMAP
    print("Performing UMAP transformation...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    point_cloud = reducer.fit_transform(data)
    
    # Cache the transformation
    print("Caching UMAP transformation...")
    with open(cache_file, 'w') as f:
        json.dump(point_cloud.tolist(), f)

# Create grid
grid = np.dstack(np.meshgrid(np.linspace(0, 1, args.nx), np.linspace(0, 1, args.ny))).reshape(-1, 2)

# Scale point cloud
point_cloud = (point_cloud - np.mean(point_cloud, axis=0)) * (np.std(grid, axis=0) / np.std(point_cloud, axis=0)) + 0.5

# Solve assignment problem
start_time = time()

cost_matrix = cdist(grid, point_cloud, 'sqeuclidean')

if args.solver == 'lapcsa':
    cost, row_ind, col_ind = lapcsa.lap(cost_matrix, min_epsilon_factor=args.min_epsilon_factor, scale_factor=args.scale_factor)
elif args.solver == 'auction':
    row_ind, col_ind, _ = auction_lap(-cost_matrix.T, args.auction_eps)
else:  # scipy
    row_ind, col_ind = linear_sum_assignment(cost_matrix.T)
    
grid_assigned = grid[col_ind]

# Save column indices to JSON
with open('column_indices.json', 'w') as f:
    json.dump(col_ind.tolist(), f)

# min_cost = cost_matrix[row_ind, col_ind].sum()
print(f"Assignment problem solved in {time() - start_time:.3f} seconds using {args.solver}")
print(f"Column indices saved to column_indices.json")

# Visualize results
plt.figure(figsize=(10, 10))

# Draw assignment lines
for start, end in zip(point_cloud, grid_assigned):
    plt.plot([start[0], end[0]], [start[1], end[1]], 
             color='gray', alpha=0.3, linewidth=1)

plt.title('Point Cloud to Grid Assignment')

# Calculate plot boundaries
points = np.vstack([point_cloud, grid_assigned])
padding = 0.2
bounds = np.array([points.min(axis=0) - padding, points.max(axis=0) + padding])
span = bounds[1] - bounds[0]

# Make plot square
if span[0] > span[1]:
    center = bounds.mean(axis=0)[1]
    bounds[:, 1] = center + span[0] * np.array([-0.5, 0.5])
else:
    center = bounds.mean(axis=0)[0]
    bounds[:, 0] = center + span[1] * np.array([-0.5, 0.5])

plt.xlim(bounds[0, 0], bounds[1, 0])
plt.ylim(bounds[0, 1], bounds[1, 1])
plt.show() 