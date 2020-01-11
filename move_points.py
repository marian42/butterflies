import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KDTree
from config import *

codes = np.load('data/latent_codes_embedded.npy')
min_value = np.min(codes, axis=0)
max_value = np.max(codes, axis=0)
codes -= (max_value + min_value) / 2
codes /= np.max(codes, axis=0)

def move(points, radius, amount):
    tree = KDTree(points)

    distances, indices = tree.query(points, k=2, return_distance=True)
    mask = distances[:, 1] < radius
    points_moved = np.count_nonzero(mask)
    indices = indices[mask, 1:]

    neighbors = points[indices[:, 0], :]
    for i in range(indices.shape[1] - 1):
        neighbors += points[indices[:, i + 1], :]
    neighbors /= indices.shape[1]
    directions = neighbors - points[mask, :]
    directions /= np.linalg.norm(directions, axis=0)

    points[mask] -= directions * amount
    return points, points_moved

RADIUS = IMAGE_SIZE / 2 / TILE_SIZE / 2**TILE_DEPTH

def move_mutiple(points, radius=RADIUS, steps=10000):
    points_moved = points.shape[0]
    i = 0
    while points_moved > 100:
        try:
            points, points_moved = move(points, radius * 2, radius / 4)
            print(points_moved)
            i += 1
        except KeyboardInterrupt:
            print("Stopping after {:d} iterations.".format(i))
    return points

codes = move_mutiple(codes)

OUTPUT_FILENAME = 'data/latent_codes_embedded_moved.npy'
min_value = np.min(codes, axis=0)
max_value = np.max(codes, axis=0)
codes -= (max_value + min_value) / 2
codes *= 0.99 / np.max(codes, axis=0)

np.save(OUTPUT_FILENAME, codes)
print("Saved to {:s}.".format(OUTPUT_FILENAME))