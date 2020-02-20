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
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import math

RADIUS = IMAGE_SIZE / TILE_SIZE / 2**TILE_DEPTH

UPDATE_RANGE = 4

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

def get_violating_range(points, range=8, verbose=False):
    distances, _ = KDTree(points).query(points, k=2, return_distance=True)
    distances = distances[:, 1]

    if verbose:
        analyze_distances(distances)

    mask = distances < RADIUS
    if range == 0:
        return mask

    violating_points = points[mask, :]
    if violating_points.shape[0] < 5:
        return mask

    distances, _ = KDTree(violating_points).query(points, k=1, return_distance=True)
    return distances[:, 0] < RADIUS * range

def move_points(points, verbose=False):
    update_mask = get_violating_range(points, range=UPDATE_RANGE, verbose=verbose)
    update_indices = np.nonzero(update_mask)[0]
    
    update_points = np.array(points[update_indices])

    if update_points.shape[0] < 5:
        return True

    voronoi = Voronoi(update_points, qhull_options='Qbb Qc Qx')

    centroids = []
    for index in tqdm(range(update_points.shape[0])) if verbose else range(update_points.shape[0]):
        region_index = voronoi.point_region[index]
        if region_index == -1:
            centroids.append(update_points[index, :])
            continue

        region = [i for i in voronoi.regions[region_index] if i != -1]
        region = region + [region[0]]
        vertices = voronoi.vertices[region]
        centroids.append(find_centroid(vertices))

    max_distance = RADIUS * 0.1

    centroids = np.array(centroids)
    if max_distance is not None:
        move_vectors = centroids - update_points
        move_distances = np.linalg.norm(move_vectors, axis=1)
        high_distances_mask = move_distances > max_distance
        move_directions = move_vectors[high_distances_mask, :] / move_distances[high_distances_mask, np.newaxis]
        move_vectors[high_distances_mask, :] = move_directions * max_distance
        points[update_indices] += move_vectors
    else:
        points[update_indices] = centroids
    return False
    
plt.figure(num=None, figsize=(10, 10), dpi=140, facecolor='w', edgecolor='k')    
last_points = None

def save_image(points, index):
    global last_points
    axes = plt.gca()
    axes.set_xlim(x_range)
    axes.set_ylim(y_range)
    violating_mask = get_violating_range(points, range=0)
    violating_range_mask = get_violating_range(points, range=UPDATE_RANGE)
    
    '''if last_points is not None:
        lines = []
        for i in range(last_points.shape[0]):
            lines.append([last_points[i, :], points[i, :]])
        lc = mc.LineCollection(lines, colors='gray', linewidths=0.4)
        axes.add_collection(lc)
    last_points = np.array(points)'''

    plt.scatter(points[violating_mask & violating_range_mask, 0], points[violating_mask, 1], s=0.3, c='red', zorder=100)
    plt.scatter(points[~violating_mask & violating_range_mask, 0], points[~violating_mask & violating_range_mask, 1], s=0.3, c='blue', zorder=100)
    plt.scatter(points[~violating_range_mask, 0], points[~violating_range_mask, 1], s=0.3, c='gray', zorder=100)

    plt.savefig('images/{:05d}.png'.format(index), bbox_inches='tight')
    plt.clf()

def process_cluster(points, cluster_label):
    try:
        steps = 0
        points = np.array(points)
        done = False
        while not done:
            done = move_points(points, verbose=False)
            steps += 1
        return points, cluster_label
    except:
        traceback.print_exc()
        return points, cluster_label

if __name__ == '__main__':
    points = np.load('data/latent_codes_embedded.npy')
    min_value = np.min(points, axis=0)
    max_value = np.max(points, axis=0)
    points -= (max_value + min_value) / 2
    points /= np.max(points, axis=0)

    x_range = (-0.54, -0.44)
    y_range = (0.14, 0.225)

    for i in range(10):
        print("Sequential step {:d}/10".format(i+1))
        move_points(points, verbose=True)

    parallel_working_mask = get_violating_range(points, range=16)
    parallel_working_set = points[parallel_working_mask, :]

    dbscan = DBSCAN(eps=RADIUS*8, min_samples=1).fit(parallel_working_set)
    cluster_labels = dbscan.labels_
    label_count = np.max(dbscan.labels_)
    print('Cluster count:', label_count)

    worker_count = os.cpu_count()
    print("Using {:d} processes.".format(worker_count))

    cluster_masks = dict()
    for cluster_label in range(label_count):
        mask = np.nonzero(cluster_labels == cluster_label)[0]
        if mask.shape[0] >= 4:
            cluster_masks[cluster_label] = mask
    
    pool = Pool(worker_count)
    progress = tqdm(total=label_count)
    
    def on_complete(args):
        moved_points, cluster_label = args
        mask = cluster_masks[cluster_label]        
        parallel_working_set[mask] = moved_points
        progress.update()

    for cluster_label in cluster_masks.keys():
        mask = cluster_masks[cluster_label]
        current_cluster_points = np.array(parallel_working_set[mask])
        pool.apply_async(process_cluster, args=(current_cluster_points, cluster_label), callback=on_complete)

    pool.close()
    pool.join()

    points[parallel_working_mask] = parallel_working_set

    done = False
    while not done:
        done = move_points(points, verbose=True)
    
    OUTPUT_FILENAME = 'data/latent_codes_embedded_moved.npy'
    min_value = np.min(points, axis=0)
    max_value = np.max(points, axis=0)
    points -= (max_value + min_value) / 2
    points *= 0.99 / np.max(points, axis=0)

    np.save(OUTPUT_FILENAME, points)
    print("Saved to {:s}.".format(OUTPUT_FILENAME))