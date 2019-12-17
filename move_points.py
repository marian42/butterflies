import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from matplotlib.patches import Circle

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

TILE_SIZE = 256
IMAGE_SIZE = 128
TILE_DEPTH = 8
RADIUS = IMAGE_SIZE / 2 / TILE_SIZE / 2**TILE_DEPTH

def create_plot(points, index):
    plt.clf
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 20)
    r = 0.25
    plt.xlim((-r, r))
    plt.ylim((-r, r))
    plt.scatter(points[:, 0], points[:, 1], marker='o', s=1, color='r', zorder=100)
    plt.savefig('data/plots/{:05d}.png'.format(i), bbox_inches='tight')
    plt.close(fig)

def move_mutiple(points, radius=RADIUS, steps=10000):
    points_moved = points.shape[0]
    i = 0
    while points_moved > 100:
        try:
            points, points_moved = move(points, radius * 2, radius / 4)

            print(points_moved)
            if i % 50 == 0:
                create_plot(points, i)
            
            i += 1
        except KeyboardInterrupt:
            print("Stopping after {:d} iterations.".format(i))
    return points

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

codes = move_mutiple(codes)

OUTPUT_FILENAME = 'data/latent_codes_embedded_moved.npy'
np.save(OUTPUT_FILENAME, moved)
prin("Saved to {:s}.".format(OUTPUT_FILENAME))