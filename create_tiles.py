from tkinter import *
import numpy as np
import json
from sklearn.cluster import KMeans
from itertools import count
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import os
from multiprocessing import Pool
import traceback
import math
import json
import csv
from config import *

latent_codes = np.load('data/latent_codes.npy')
codes = np.load('data/latent_codes_embedded_moved.npy')

from image_loader import ImageDataset
dataset = ImageDataset()

codes_by_depth = []
hashes_by_depth = []

rotation_file = open('data/rotations_calculated.csv', 'r')
reader = csv.reader(rotation_file)
rotations = {row[0]: float(row[1]) for row in reader}
rotation_file.close()

def create_tile(depth, x, y):
    tile_file_name = TILE_FILE_FORMAT.format(depth + DEPTH_OFFSET, x, y)
    if os.path.exists(tile_file_name):
        return
    
    tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (255, 255, 255))
    is_empty = True

    if depth < TILE_DEPTH:
        for a in range(2):
            for b in range(2):
                old_tile_file_name = TILE_FILE_FORMAT.format(depth + 1 + DEPTH_OFFSET, x * 2 + a, y * 2 + b)
                if not os.path.exists(old_tile_file_name):
                    continue
                image = Image.open(old_tile_file_name)
                image = image.resize((TILE_SIZE // 2, TILE_SIZE // 2), resample=Image.BICUBIC)
                tile.paste(image, (a * TILE_SIZE // 2, b * TILE_SIZE // 2))
                is_empty = False

    if depth > 1:
        margin = (IMAGE_SIZE / 2 + SHADOW_RADIUS) / TILE_SIZE
        x_range = ((x - margin) / 2**depth, (x + 1 + margin) / 2**depth)
        y_range = ((y - margin) / 2**depth, (y + 1 + margin) / 2**depth)

        codes_current = codes_by_depth[depth]
        hashes = hashes_by_depth[depth]

        mask = (codes_current[:, 0] > x_range[0]) \
            & (codes_current[:, 0] < x_range[1]) \
            & (codes_current[:, 1] > y_range[0]) \
            & (codes_current[:, 1] < y_range[1])
        indices = mask.nonzero()[0]

        if indices.shape[0] > 0:
            is_empty = False
            positions = codes_current[indices, :]
            positions *= 2**depth * TILE_SIZE
            positions -= np.array((x * TILE_SIZE, y * TILE_SIZE))[np.newaxis, :]

            for i in range(indices.shape[0]):
                index = indices[i]
                image_id = hashes[index]
                angle = rotations[image_id]
                image_file_name = 'data/images_alpha/{:s}.png'.format(image_id)
                image_original = Image.open(image_file_name)
                image = image_original.rotate(angle, resample=Image.BICUBIC, expand=True)
                size = int(IMAGE_SIZE * image.size[0] / image_original.size[0])
                image = image.resize((size, size), resample=Image.BICUBIC)

                shadow_mask = Image.new("L", (size + 2 * SHADOW_RADIUS, size + 2 * SHADOW_RADIUS), 0)
                shadow_mask.paste(image.split()[-1], (SHADOW_RADIUS, SHADOW_RADIUS))
                shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(radius=SHADOW_RADIUS // 2))
                enhancer = ImageEnhance.Brightness(shadow_mask)
                shadow_mask = enhancer.enhance(SHADOW_VALUE)

                tile.paste((0, 0, 0), (int(positions[i, 0] - size // 2 - SHADOW_RADIUS), int(positions[i, 1] - size // 2 - SHADOW_RADIUS)), mask=shadow_mask)
                tile.paste(image, (int(positions[i, 0] - size // 2), int(positions[i, 1] - size // 2)), mask=image)
    
    if not is_empty:
        tile.save(tile_file_name)

def try_create_tile(*args):
    try:
        create_tile(*args)
    except:
        traceback.print_exc()

def kmeans(points, points_latent_codes, n):
    if n == 0:
        return [], np.zeros((0, 2))
    if points.shape[0] <= n:
        return range(points.shape[0]), points
    kmeans = KMeans(n_clusters=n)
    kmeans_clusters = kmeans.fit_predict(points)
    result_indices = []
    for i in range(n):
        cluster_indices = np.nonzero(kmeans_clusters == i)[0]
        dist = np.linalg.norm(points_latent_codes[cluster_indices] - np.mean(points_latent_codes[cluster_indices], axis=0), axis=1)
        result_indices.append(cluster_indices[np.argmin(dist)])
    return result_indices, kmeans.cluster_centers_

def get_kmeans(count, subdivisions):
    if subdivisions == 1:
        return kmeans(codes, latent_codes, count)
        return np.array(list(), dtype=int)
    
    result_indices = []
    result_points = []
    for x in tqdm(range(subdivisions)):
        for y in range(subdivisions):
            x_range = (-1 + 2 * x / subdivisions, -1 + 2 * (x + 1) / subdivisions)
            y_range = (-1 + 2 * y / subdivisions, -1 + 2 * (y + 1) / subdivisions)

            mask = (codes[:, 0] > x_range[0]) \
                & (codes[:, 0] <= x_range[1]) \
                & (codes[:, 1] > y_range[0]) \
                & (codes[:, 1] <= y_range[1])
            indices = np.nonzero(mask)[0]
            codes_mask = codes[mask, :]
            kmeans_indices, kmeans_points = kmeans(codes_mask, latent_codes[mask, :], int(count * indices.shape[0] / codes.shape[0]))
            for i in kmeans_indices:
                result_indices.append(indices[i])
            result_points.append(kmeans_points)
    
    return result_indices, np.concatenate(result_points)

for depth in range(TILE_DEPTH):
    print("Running k-means for depth {:d}.".format(depth))
    number_of_items = 2**(2*depth) * 2
    indices, points = get_kmeans(number_of_items, max(1, 2**(depth - 2)))
    codes_by_depth.append(points)
    hashes_by_depth.append([dataset.hashes[i] for i in indices])

json_dict = {depth + DEPTH_OFFSET: [{'image': hash, 'x': codes_by_depth[depth][i, 0], 'y': codes_by_depth[depth][i, 1]} for i, hash in enumerate(hashes)] for depth, hashes in enumerate(hashes_by_depth)}
json_string = json.dumps(json_dict)
with open('data/clusters.json', 'w') as file:
    file.write(json_string)


codes_by_depth.append(codes)
hashes_by_depth.append(dataset.hashes)

worker_count = os.cpu_count()
print("Using {:d} processes.".format(worker_count))

for depth in range(TILE_DEPTH, -4, -1):
    pool = Pool(worker_count)
    progress = tqdm(total=(2**(2 * depth + 2)), desc='Depth {:d}'.format(depth + DEPTH_OFFSET))

    def on_complete(*_):
        progress.update()

    for x in range(math.floor(-2**depth), math.ceil(2**depth)):
        tile_directory = os.path.dirname(TILE_FILE_FORMAT.format(depth + DEPTH_OFFSET, x, 0))
        if not os.path.exists(tile_directory):
            os.makedirs(tile_directory)
        for y in range(math.floor(-2**depth), math.ceil(2**depth)):
            pool.apply_async(try_create_tile, args=(depth, x, y), callback=on_complete)
    pool.close()
    pool.join()