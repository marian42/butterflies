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

RADIUS = IMAGE_SIZE / 2 / TILE_SIZE / 2**TILE_DEPTH

VELOCITY_DECAY = 0.8
REPELL_FORCE_STRENGTH = 0.5
RESET_FORCE_STRENGTH = 0.2 * RADIUS

def move_points(points, verbose=False, max_iter=1000):
    velocity = np.zeros(points.shape)
    original_points = np.array(points)

    is_finalizing_phase = False
    finalizing_steps_left = 40

    for step in tqdm(range(max_iter)) if verbose else range(max_iter):
        next_points = points + velocity

        distances, indices = KDTree(next_points).query(next_points, k=5, return_distance=True)
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                distance = distances[i, j]
                if distance > 2 * RADIUS:
                    if j == 0:
                        velocity[i, :] *= 0.5
                    break
                direction = next_points[indices[i, j], :] - next_points[i, :]
                direction /= distance
                overlap = 2 * RADIUS - distance
                force = min(REPELL_FORCE_STRENGTH, step / 50) * overlap
                velocity[i, :] -= direction * force

        if not is_finalizing_phase:
            reset_direction = points - original_points
            non_zero = reset_direction != 0
            reset_direction[non_zero] /= np.linalg.norm(reset_direction[non_zero], axis=0)
            velocity -= reset_direction * RESET_FORCE_STRENGTH
        
        points += velocity
        
        velocity *= VELOCITY_DECAY

        if not is_finalizing_phase and step % 10 == 0:
            distances, _ = KDTree(points).query(points, k=2, return_distance=True)
            if verbose:
                tqdm.write('Violating points: {:d}'.format(np.count_nonzero(distances[:, 1] < 2 * RADIUS)))
            if np.count_nonzero(distances[:, 1] < 2 * RADIUS * 0.85) < points.shape[0] * 0.01:
                is_finalizing_phase = True

        if is_finalizing_phase:
            finalizing_steps_left -= 1
            if finalizing_steps_left == 0:
                break

def wiggle_duplicates(points):
    while True:
        distances, indices = KDTree(points).query(points, k=2, return_distance=True)
        mask = distances[:, 1] == 0
        indices = indices[mask, 1]
        if indices.shape[0] == 0:
            return
        print("Found {:d} duplicates.".format(indices.shape[0]))
        points[indices, :] += np.random.normal(scale=1e-9, size=(indices.shape[0], 2))

def get_violating_range(points, range=8):
    distances, _ = KDTree(points).query(points, k=2, return_distance=True)
    distances = distances[:, 1]
    mask = distances < RADIUS * 2

    if range == 0:
        return mask

    violating_points = points[mask, :]

    distances, _ = KDTree(violating_points).query(points, k=1, return_distance=True)
    return distances[:, 0] < RADIUS * 2 * range

def process_cluster(points, cluster_label):
    try:
        move_points(points)
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

    wiggle_duplicates(points)

    move_points(points, verbose=True, max_iter=40)

    parallel_working_mask = get_violating_range(points, range=8)
    parallel_working_set = points[parallel_working_mask, :]

    dbscan = DBSCAN(eps=RADIUS * 16, min_samples=1).fit(parallel_working_set)
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
    if any(remainder):
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

    move_points(points, verbose=True, max_iter=21)
    
    OUTPUT_FILENAME = 'data/latent_codes_embedded_moved.npy'
    min_value = np.min(points, axis=0)
    max_value = np.max(points, axis=0)
    points -= (max_value + min_value) / 2
    points *= 0.99 / np.max(points, axis=0)

    np.save(OUTPUT_FILENAME, points)
    print("Saved to {:s}.".format(OUTPUT_FILENAME))