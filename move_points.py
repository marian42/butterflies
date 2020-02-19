import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KDTree
from config import *
from multiprocessing import Pool
import os
import traceback
import time

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

def move_mutiple(points, radius=RADIUS, max_points_in_radius=100, verbose=True):
    points_moved = points.shape[0]
    i = 0
    start_time = time.time()
    while points_moved > max_points_in_radius:
        points, points_moved = move(points, radius * 2, radius / 4)
        if i % 100 == 0:
            if verbose:
                print(points.shape[0], points_moved, '/', max_points_in_radius)
            elif time.time() - start_time > 120:
                verbose = True
        i += 1
    return points

def process_cluster(cluster_label):
    try:
        mask = cluster_labels == cluster_label
        label_codes = np.array(codes[mask])
        label_codes = move_mutiple(label_codes, max_points_in_radius=min(20, label_codes.shape[0] // 10), verbose=False)
        return cluster_label, label_codes
    except:
        traceback.print_exc()

if __name__ == '__main__':
    codes = np.load('data/latent_codes_embedded.npy')
    min_value = np.min(codes, axis=0)
    max_value = np.max(codes, axis=0)
    codes -= (max_value + min_value) / 2
    codes /= np.max(codes, axis=0)

    print('Clustering...')

    from sklearn.cluster import DBSCAN

    original = np.array(codes)

    dbscan = DBSCAN(eps=RADIUS*4, min_samples=1).fit(codes)
    cluster_labels = dbscan.labels_
    label_count = np.max(dbscan.labels_)
    print('Label count:', label_count)

    worker_count = os.cpu_count()
    print("Using {:d} processes.".format(worker_count))


    masks = dict()
    for cluster_label in tqdm(range(label_count)):
        mask = np.nonzero(cluster_labels == cluster_label)[0]
        masks[cluster_label] = mask

    pool = Pool(worker_count)
    progress = tqdm(total=label_count)
    
    def on_complete(args):
        cluster_label, moved_codes = args
        mask = masks[cluster_label]
        codes[mask] = moved_codes
        progress.update()

    for cluster_label in range(label_count):
        mask = masks[cluster_label]
        label_codes = np.array(codes[mask])
        if label_codes.shape[0] < 2:
            progress.update()
            continue
        pool.apply_async(process_cluster, args=(cluster_label,), callback=on_complete)

    pool.close()
    pool.join()

    print(np.count_nonzero(original != codes))

    print("Moving full dataset.")
    codes = move_mutiple(codes)

    OUTPUT_FILENAME = 'data/latent_codes_embedded_moved.npy'
    min_value = np.min(codes, axis=0)
    max_value = np.max(codes, axis=0)
    codes -= (max_value + min_value) / 2
    codes *= 0.99 / np.max(codes, axis=0)

    np.save(OUTPUT_FILENAME, codes)
    print("Saved to {:s}.".format(OUTPUT_FILENAME))