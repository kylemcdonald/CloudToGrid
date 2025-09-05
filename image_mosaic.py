import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from umap import UMAP
from lap import lapjv
from scipy.spatial.distance import cdist

def load_and_resize_images(directory, target_size, required_images=None):
    """Load and resize images from directory."""
    image_files = []
    images = []
    
    # Get all image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [f for f in os.listdir(directory) 
             if os.path.splitext(f.lower())[1] in valid_extensions]
    
    for filename in files:
        filepath = os.path.join(directory, filename)
        try:
            img = Image.open(filepath)
            img = img.convert('RGB')
            img = img.resize(target_size, Image.LANCZOS)
            img_array = np.array(img)
            
            images.append(img_array)
            image_files.append(filename)
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    # If we need more images than we have, pad with black images
    if required_images and len(images) < required_images:
        black_image = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        while len(images) < required_images:
            images.append(black_image)
            image_files.append(f"blank_{len(images)}.png")
    
    return np.array(images), image_files

def create_mosaic(images, nx, ny):
    """Create a mosaic from images."""
    if len(images) != nx * ny:
        raise ValueError(f"Number of images ({len(images)}) must equal nx*ny ({nx*ny})")
    
    h, w, c = images[0].shape
    mosaic = np.zeros((h*ny, w*nx, c), dtype=np.uint8)
    
    for idx, image in enumerate(images):
        i = idx // nx
        j = idx % nx
        mosaic[i*h:(i+1)*h, j*w:(j+1)*w] = image
    
    return mosaic

def main():
    parser = argparse.ArgumentParser(description='Create image mosaic with UMAP embedding')
    parser.add_argument('input_dir', help='Input directory containing images')
    parser.add_argument('output_dir', help='Output directory for results')
    parser.add_argument('--size', type=int, default=32, help='Size to resize images to')
    parser.add_argument('--grid-height', type=int, default=32, help='Height of output grid (ignored if max-images is set)')
    parser.add_argument('--max-images', type=int, help='Maximum number of images to process')
    
    args = parser.parse_args()
    
    # Calculate grid dimensions
    if args.max_images:
        # Calculate grid dimensions based on max_images
        grid_height = int(np.sqrt(args.max_images / 2))
        grid_width = grid_height * 2
    else:
        grid_height = args.grid_height
        grid_width = grid_height * 2
    
    required_images = grid_width * grid_height
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and resize images
    target_size = (args.size, args.size)
    images, filenames = load_and_resize_images(args.input_dir, target_size, required_images)
    
    if args.max_images:
        images = images[:args.max_images]
        filenames = filenames[:args.max_images]
    
    # Flatten images for UMAP
    flattened_images = images.reshape(len(images), -1)
    
    # Generate UMAP embedding
    umap = UMAP(n_components=2, random_state=42)
    embedding = umap.fit_transform(flattened_images)
    
    # Normalize embedding to [0,1] range
    embedding_normalized = (embedding - embedding.min(axis=0)) / (embedding.max(axis=0) - embedding.min(axis=0))
    
    # Create grid with 2:1 aspect ratio
    xv, yv = np.meshgrid(np.linspace(0, 1, grid_width), np.linspace(0, 1, grid_height))
    grid = np.dstack((xv, yv)).reshape(-1, 2)
    
    # Calculate optimal assignment
    cost = cdist(grid, embedding_normalized, 'sqeuclidean')
    cost = cost * (100000. / cost.max())
    cost = cost.astype(int)
    
    # Add extend_cost=True to handle non-square matrices
    _, _, col_assigns = lapjv(cost, extend_cost=True)
    grid_assignments = grid[col_assigns]
    
    # Save UMAP positions
    umap_positions = {
        filename: {"x": float(pos[0]), "y": float(pos[1])} 
        for filename, pos in zip(filenames, embedding_normalized)
    }
    
    with open(os.path.join(args.output_dir, 'umap_positions.json'), 'w') as f:
        json.dump(umap_positions, f, indent=2)
    
    # Save grid positions
    grid_positions = {
        filename: {"x": float(pos[0]), "y": float(pos[1])} 
        for filename, pos in zip(filenames, grid_assignments)
    }
    
    with open(os.path.join(args.output_dir, 'grid_positions.json'), 'w') as f:
        json.dump(grid_positions, f, indent=2)
    
    # Create and save mosaic with 2:1 aspect ratio
    grid_tuples = [(y,x) for (x,y) in map(tuple, grid_assignments)]
    sorted_indices = np.lexsort(([t[1] for t in grid_tuples], [t[0] for t in grid_tuples]))
    sorted_images = [images[i] for i in sorted_indices]
    
    mosaic = create_mosaic(sorted_images, grid_width, grid_height)
    
    mosaic_image = Image.fromarray(mosaic)
    mosaic_image.save(os.path.join(args.output_dir, 'mosaic.png'))

if __name__ == '__main__':
    main() 