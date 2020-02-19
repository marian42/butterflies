import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KDTree
from scipy.spatial import Voronoi
from config import *
from multiprocessing import Pool
import os
import traceback
import time
from matplotlib import pyplot as plt
from matplotlib import collections  as mc

points = np.load('data/latent_codes_embedded.npy')
min_value = np.min(points, axis=0)
max_value = np.max(points, axis=0)
points -= (max_value + min_value) / 2
points /= np.max(points, axis=0)


x_range = (-1, 1)
y_range = (-1, 1)

#mask = (points[:, 0] > x_range[0]) & (points[:, 0] < x_range[1]) & (points[:, 1] > y_range[0]) & (points[:, 1] < y_range[1])
#points = np.array(points[mask, :])

update_steps = np.zeros(points.shape[0])


RADIUS = IMAGE_SIZE / 2 / TILE_SIZE / 2**TILE_DEPTH

def find_centroid(vertices):
    area = 0
    centroid_x = 0
    centroid_y = 0
    for i in range(len(vertices)-1):
      step = (vertices[i  , 0] * vertices[i+1, 1]) - \
             (vertices[i+1, 0] * vertices[i  , 1])
      area += step
      centroid_x += (vertices[i, 0] + vertices[i+1, 0]) * step
      centroid_y += (vertices[i, 1] + vertices[i+1, 1]) * step
    area /= 2
    # prevent division by zero - equation linked above
    if area == 0:
        area += 0.0000001
    centroid_x = (1.0 / (6.0*area)) * centroid_x
    centroid_y = (1.0 / (6.0*area)) * centroid_y
    return [centroid_x, centroid_y]

def analyze_distances(distances):
    result = 'smallest relative distance: {:f}, '.format(np.min(distances) / RADIUS)
    N = 5
    thresholds = [v / N for v in range(1, N + 1)]
    result += ', '.join(['{:d} below {:.1f}'.format(np.count_nonzero(distances < RADIUS * v), v) for v in thresholds])
    print(result)

def get_update_mask(points, multiplier=8):
    distances, _ = KDTree(points).query(points, k=2, return_distance=True)
    distances = distances[:, 1]
    mask = distances < RADIUS
    analyze_distances(distances)

    distances, _ = KDTree(points[distances < RADIUS, :]).query(points, k=1, return_distance=True)
    mask = distances[:, 0] < RADIUS * (1 + update_steps * 0.5)
    return mask

def move_points():
    update_mask = get_update_mask(points)
    update_indices = np.nonzero(update_mask)[0]
    update_steps[update_indices] += 1
    
    voronoi = Voronoi(points, qhull_options='Qbb Qc Qx')

    centroids = []
    for index in tqdm(update_indices):
        region_index = voronoi.point_region[index]
        if region_index == -1:
            centroids.append(points[index, :])
            continue

        region = [i for i in voronoi.regions[region_index] if i != -1]
        region = region + [region[0]]
        vertices = voronoi.vertices[region]
        centroids.append(find_centroid(vertices))
    
    points[update_indices] = np.array(centroids)
    
plt.figure(num=None, figsize=(40, 40), dpi=100, facecolor='w', edgecolor='k')    
last_points = None

def get_violating_mask(points):
    distances, _ = KDTree(points).query(points, k=2, return_distance=True)
    distances = distances[:, 1]
    mask = distances < RADIUS
    return mask

def save_image(points, index):
    global last_points
    axes = plt.gca()
    axes.set_xlim(x_range)
    axes.set_ylim(y_range)
    violating_mask = get_violating_mask(points)
    print(np.count_nonzero(violating_mask))
    
    '''if last_points is not None:
        lines = []
        for i in range(last_points.shape[0]):
            lines.append([last_points[i, :], points[i, :]])
        lc = mc.LineCollection(lines, colors='gray', linewidths=0.4)
        axes.add_collection(lc)
    last_points = np.array(points)'''

    plt.scatter(points[violating_mask, 0], points[violating_mask, 1], s=0.3, c='red', zorder=100)
    plt.scatter(points[~violating_mask, 0], points[~violating_mask, 1], s=0.3, c='blue', zorder=100)

    plt.savefig('images/{:05d}.png'.format(index), bbox_inches='tight')
    plt.clf()

save_image(points, 0)

for i in range(500):
    move_points()
    #save_image(points, i+1)
