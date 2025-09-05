import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
import math
from PIL import Image
import json

# Set random seed for reproducibility
np.random.seed(42)

# Load MNIST data
print("Loading MNIST data...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0  # Normalize to [0,1]
X, y = shuffle(X, y, random_state=42)  # Shuffle the data

# Define grid dimensions
grid_size = 244
total_images = grid_size * grid_size
print(f"Creating {grid_size}x{grid_size} grid ({total_images} images total)")

# Pad or truncate dataset to match grid size
if len(X) < total_images:
    # Pad by repeating images if needed
    repeat_count = math.ceil(total_images / len(X))
    X = np.tile(X, (repeat_count, 1))[:total_images]
else:
    # Take only the number of images we need
    X = X[:total_images]

# Reshape each 28x28 image
image_size = 28
images = X.reshape(-1, image_size, image_size)

# Create the full grid image
grid_pixels = grid_size * image_size
full_image = np.zeros((grid_pixels, grid_pixels))

# Fill the grid with images
print("Creating grid visualization...")
for idx in range(total_images):
    i = idx // grid_size  # row
    j = idx % grid_size   # column
    
    # Calculate position in the full image
    row_start = i * image_size
    row_end = (i + 1) * image_size
    col_start = j * image_size
    col_end = (j + 1) * image_size
    
    # Place the image
    full_image[row_start:row_end, col_start:col_end] = images[idx]

# Convert to 8-bit grayscale and save
print("Saving visualization...")
image_data = (full_image * 255).astype(np.uint8)
image = Image.fromarray(image_data, mode='L')
image.save('mnist_grid_6832x6832.png', optimize=False)

# Save raw image data as JSON
print("Saving raw image data to JSON...")
json_data = []
for idx in range(total_images):
    # Convert the image data to uint8 and flatten to a list
    raw_image = (images[idx] * 255).astype(np.uint8).flatten().tolist()
    json_data.append(raw_image)

with open('mnist_data.json', 'w') as f:
    json.dump(json_data, f)

print("Done! Image saved as 'mnist_grid_6832x6832.png' and raw data saved as 'mnist_data.json'") 