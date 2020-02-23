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
        step = (vertices[i, 0] * vertices[i+1, 1]) - (vertices[i+1, 0] * vertices[i, 1])
        area += step
        centroid_x += (vertices[i, 0] + vertices[i+1, 0]) * step
        centroid_y += (vertices[i, 1] + vertices[i+1, 1]) * step
    area /= 2
    if area == 0:
        area = 0.0000001
    centroid_x *= 1.0 / (6.0*area)
    centroid_y *= 1.0 / (6.0*area)
    return [centroid_x, centroid_y], abs(area)

def analyze_distances(distances):
    result = 'smallest relative distance: {:f}, '.format(np.min(distances) / RADIUS)
    N = 5
    thresholds = [v / N for v in range(1, N + 1)]
    result += ', '.join(['{:d} below {:.1f}'.format(np.count_nonzero(distances < RADIUS * v), v) for v in thresholds])
    print(result)

def get_violating_range(points, range=8):
    distances, _ = KDTree(points).query(points, k=2, return_distance=True)
    distances = distances[:, 1]
    mask = distances < RADIUS

    if range == 0:
        return mask

    violating_points = points[mask, :]
    if violating_points.shape[0] < 5:
        return mask

    distances, _ = KDTree(violating_points).query(points, k=1, return_distance=True)
    return distances[:, 0] < RADIUS * range

def check_result(points, fraction=0.01, threshold=0.9):
    distances, _ = KDTree(points).query(points, k=2, return_distance=True)
    distances = distances[:, 1]
    return np.count_nonzero(distances < RADIUS * threshold) < fraction * points.shape[0]

def move_points(points, verbose=False, max_iter=1000):
    for iteration in tqdm(range(max_iter)) if verbose else range(max_iter):
        voronoi = Voronoi(points, qhull_options='Qbb Qc Qx')
        result = []
        for index in range(points.shape[0]):
            region_index = voronoi.point_region[index]
            region = [i for i in voronoi.regions[region_index] if i != -1]
            region = region + [region[0]]
            vertices = voronoi.vertices[region]
            centroid, area = find_centroid(vertices)
            is_edge = area > math.pi * RADIUS**2
            if not is_edge:
                result.append(centroid)
            else:
                result.append(points[index, :])

        points = np.array(result)
        iteration += 1

        if iteration % 40 == 0 and check_result(points):
            break
        if verbose:
            tqdm.write('Violating points: {:d}'.format(np.count_nonzero(get_violating_range(points, range=0))))
    return points
    
plt.figure(num=None, figsize=(20, 20), dpi=140, facecolor='w', edgecolor='k')    
last_points = None

def save_image(points, index, voronoi=None, edge_points=None):
    global last_points
    axes = plt.gca()
    x = points[:, 0]
    y = points[:, 1]
    violating_mask = get_violating_range(points, range=0)

    plt.scatter(points[violating_mask, 0], points[violating_mask, 1], s=0.3, c='red', zorder=100)
    plt.scatter(points[~violating_mask, 0], points[~violating_mask, 1], s=0.3, c='blue', zorder=100)
    if edge_points is not None:
        plt.scatter(edge_points[:, 0], edge_points[:, 1], s=0.3, c='green', zorder=1000)

    if voronoi is not None:
        lines = []
        for a, b in voronoi.ridge_vertices:
            if a == -1 or b == -1:
                continue            
            lines.append([voronoi.vertices[a, :], voronoi.vertices[b, :]])
        lc = mc.LineCollection(lines, colors='gray', linewidths=0.2)
        axes.add_collection(lc)
        last_points = np.array(points)

    plt.savefig('images/{:05d}.png'.format(index), bbox_inches='tight')
    plt.clf()

def process_cluster(points, cluster_label):
    try:
        return move_points(points), cluster_label
    except:
        traceback.print_exc()
    return points, cluster_label

if __name__ == '__main__':
    points = np.load('data/latent_codes_embedded.npy')
    min_value = np.min(points, axis=0)
    max_value = np.max(points, axis=0)
    points -= (max_value + min_value) / 2
    points /= np.max(points, axis=0)

    points = move_points(points, verbose=True, max_iter=10)

    parallel_working_mask = get_violating_range(points, range=4)
    parallel_working_set = points[parallel_working_mask, :]

    dbscan = DBSCAN(eps=RADIUS * 8, min_samples=1).fit(parallel_working_set)
    cluster_labels = dbscan.labels_
    label_count = np.max(dbscan.labels_)
    print('Cluster count:', label_count)
    
    cluster_masks = dict()
    remainder = []
    for cluster_label in range(label_count):
        mask = np.nonzero(cluster_labels == cluster_label)[0]
        if mask.shape[0] >= 20:
            cluster_masks[cluster_label] = mask
        else:
            remainder.append(mask)
    cluster_masks[-1] = np.concatenate(remainder)
    
    worker_count = os.cpu_count()
    pool = Pool(worker_count)
    cluster_labels = sorted(cluster_masks.keys(), key=lambda label: cluster_masks[label].shape[0], reverse=True)

    print('The largest cluster contains {:d} points (should be < 100k)'.format(cluster_masks[cluster_labels[0]].shape[0]))
    print("Using {:d} processes.".format(worker_count))
    
    progress = tqdm(total=sum(cluster_masks[label].shape[0] for label in cluster_masks.keys()))
    
    def on_complete(args):
        moved_points, cluster_label = args
        mask = cluster_masks[cluster_label]
        parallel_working_set[mask] = moved_points
        progress.update(mask.shape[0])

    for cluster_label in cluster_labels:
        mask = cluster_masks[cluster_label]
        current_cluster_points = np.array(parallel_working_set[mask])
        pool.apply_async(process_cluster, args=(current_cluster_points, cluster_label), callback=on_complete)

    pool.close()
    pool.join()

    points[parallel_working_mask] = parallel_working_set

    points = move_points(points, verbose=True, max_iter=10)
    
    OUTPUT_FILENAME = 'data/latent_codes_embedded_moved.npy'
    min_value = np.min(points, axis=0)
    max_value = np.max(points, axis=0)
    points -= (max_value + min_value) / 2
    points *= 0.99 / np.max(points, axis=0)

    np.save(OUTPUT_FILENAME, points)
    print("Saved to {:s}.".format(OUTPUT_FILENAME))