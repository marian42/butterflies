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
import random

def create_tile(depth, x, y):
    tile_file_name = TILE_FILE_FORMAT.format('', depth + DEPTH_OFFSET, x, y)
    tile_file_name_hq = TILE_FILE_FORMAT.format('@2x', depth + DEPTH_OFFSET, x, y)
    if os.path.exists(tile_file_name) and (not CREATE_HQ_TILES or os.path.exists(CREATE_HQ_TILES)):
        return
    
    tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (255, 255, 255))
    tile_hq = Image.new("RGB", (TILE_SIZE * 2, TILE_SIZE * 2), (255, 255, 255))
    is_empty = True

    if depth < TILE_DEPTH:
        for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
            old_tile_file_name = TILE_FILE_FORMAT.format('', depth + 1 + DEPTH_OFFSET, x * 2 + a, y * 2 + b)
            if os.path.exists(old_tile_file_name):
                image = Image.open(old_tile_file_name)
                image = image.resize((TILE_SIZE // 2, TILE_SIZE // 2), resample=Image.BICUBIC)
                tile.paste(image, (a * TILE_SIZE // 2, b * TILE_SIZE // 2))
                is_empty = False

            old_tile_file_name_hq = TILE_FILE_FORMAT.format('@2x', depth + 1 + DEPTH_OFFSET, x * 2 + a, y * 2 + b)
            if CREATE_HQ_TILES and os.path.exists(old_tile_file_name_hq):
                image = Image.open(old_tile_file_name)
                image = image.resize((TILE_SIZE, TILE_SIZE), resample=Image.BICUBIC)
                tile_hq.paste(image, (a * TILE_SIZE, b * TILE_SIZE))
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
                image_rotated = image_original.rotate(angle, resample=Image.BICUBIC, expand=True)
                size = int(IMAGE_SIZE * image_rotated.size[0] / image_original.size[0])
                image = image_rotated.resize((size, size), resample=Image.BICUBIC)
                
                image_hq = image_rotated.resize((size * 2, size * 2), resample=Image.BICUBIC)
                shadow_mask_hq = Image.new("L", (size * 2 + 2 * SHADOW_RADIUS * 2, size * 2 + 2 * SHADOW_RADIUS * 2), 0)
                shadow_mask_hq.paste(image_hq.split()[-1], (SHADOW_RADIUS * 2, SHADOW_RADIUS * 2))
                shadow_mask_hq = shadow_mask_hq.filter(ImageFilter.GaussianBlur(radius=SHADOW_RADIUS))
                enhancer = ImageEnhance.Brightness(shadow_mask_hq)
                shadow_mask_hq = enhancer.enhance(SHADOW_VALUE)

                shadow_mask = shadow_mask_hq.resize((shadow_mask_hq.size[0] // 2, shadow_mask_hq.size[1] // 2))

                tile.paste((0, 0, 0), (int(positions[i, 0] - size / 2 - SHADOW_RADIUS), int(positions[i, 1] - size / 2 - SHADOW_RADIUS)), mask=shadow_mask)
                tile.paste(image, (int(positions[i, 0] - size / 2), int(positions[i, 1] - size / 2)), mask=image)

                if CREATE_HQ_TILES:
                    tile_hq.paste((0, 0, 0), (int(positions[i, 0] * 2 - size - SHADOW_RADIUS * 2), int(positions[i, 1] * 2 - size - SHADOW_RADIUS * 2)), mask=shadow_mask_hq)
                    tile_hq.paste(image_hq, (int(positions[i, 0] * 2 - size), int(positions[i, 1] * 2 - size)), mask=image_hq)

        
    if not is_empty:
        tile.save(tile_file_name, quality=TILE_IMAGE_QUALITY)
        if CREATE_HQ_TILES:
            tile_hq.save(tile_file_name_hq, quality=TILE_IMAGE_QUALITY)

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
        if cluster_indices.shape[0] == 0:
            # k-Means creates empty clusters when the dataset contains less than n *distinct* points, but more than n *total* points (due to duplicates)
            return range(n), points[:n, :]
        dist = np.linalg.norm(points_latent_codes[cluster_indices] - np.mean(points_latent_codes[cluster_indices], axis=0), axis=1)
        result_indices.append(cluster_indices[np.argmin(dist)])
    return result_indices, kmeans.cluster_centers_


def get_kmeans(depth, x, y):
    try:
        number_of_items = 2**(2*depth) * 2
        subdivisions = max(1, 2**(depth - 2))

        if subdivisions == 1:
            kmeans_indices, kmeans_points = kmeans(codes, latent_codes, number_of_items)
            return depth, kmeans_indices, kmeans_points

        x_range = (-1 + 2 * x / subdivisions, -1 + 2 * (x + 1) / subdivisions)
        y_range = (-1 + 2 * y / subdivisions, -1 + 2 * (y + 1) / subdivisions)

        mask = (codes[:, 0] > x_range[0]) \
            & (codes[:, 0] <= x_range[1]) \
            & (codes[:, 1] > y_range[0]) \
            & (codes[:, 1] <= y_range[1])
        indices = np.nonzero(mask)[0]
        codes_mask = codes[mask, :]
        kmeans_indices, kmeans_points = kmeans(codes_mask, latent_codes[mask, :], int(number_of_items * indices.shape[0] / codes.shape[0]))

        return depth, indices[kmeans_indices], kmeans_points
    except:
        traceback.print_exc()

if __name__ == '__main__':
    latent_codes = np.load('data/latent_codes.npy')
    codes = np.load(LATENT_CODES_EMBEDDED_MOVED_FILE_NAME)

    from image_loader import ImageDataset
    dataset = ImageDataset()

    rotation_file = open(ROTATIONS_CALCULATED_FILENAME, 'r')
    reader = csv.reader(rotation_file)
    rotations = {row[0]: float(row[1]) for row in reader}
    rotation_file.close()

    kmeans_tasks = []
    for depth in range(TILE_DEPTH):
        subdivisions = max(1, 2**(depth - 2))
        for x in range(subdivisions):
            for y in range(subdivisions):
                kmeans_tasks.append((depth, x, y))

    worker_count = os.cpu_count()
    print("Using {:d} processes.".format(worker_count))
    pool = Pool(worker_count)
    progress = tqdm(total=len(kmeans_tasks), desc='Running k-means')

    codes_by_depth = [[] for _ in range(TILE_DEPTH)]
    hashes_by_depth = [[] for _ in range(TILE_DEPTH)]

    def on_complete(args):
        depth, kmeans_indices, kmeans_points = args
        codes_by_depth[depth].append(kmeans_points)
        for i in kmeans_indices:
            hashes_by_depth[depth].append(dataset.hashes[i])
        progress.update()

    for depth, x, y in kmeans_tasks:
        pool.apply_async(get_kmeans, args=(depth, x, y), callback=on_complete)
    
    pool.close()
    pool.join()

    for depth in range(TILE_DEPTH):
        codes_by_depth[depth] = np.concatenate(codes_by_depth[depth])
    
    json_dict = {depth + DEPTH_OFFSET: [{'image': hash, 'x': codes_by_depth[depth][i, 0], 'y': codes_by_depth[depth][i, 1]} for i, hash in enumerate(hashes)] for depth, hashes in enumerate(hashes_by_depth)}
    json_string = json.dumps(json_dict)
    with open('data/clusters.json', 'w') as file:
        file.write(json_string)

    codes_by_depth.append(codes)
    hashes_by_depth.append(dataset.hashes)
    
    print("Using {:d} processes.".format(worker_count))

    for depth in range(TILE_DEPTH, -4, -1):
        pool = Pool(worker_count)
        progress = tqdm(total=(2**(2 * depth + 2)), desc='Depth {:d}'.format(depth + DEPTH_OFFSET))

        def on_complete(*_):
            progress.update()

        tile_addresses = []

        for x in range(math.floor(-2**depth), math.ceil(2**depth)):
            tile_directory = os.path.dirname(TILE_FILE_FORMAT.format('', depth + DEPTH_OFFSET, x, 0))
            if not os.path.exists(tile_directory):
                os.makedirs(tile_directory)
            tile_directory_hq = os.path.dirname(TILE_FILE_FORMAT.format('@2x', depth + DEPTH_OFFSET, x, 0))
            if not os.path.exists(tile_directory_hq):
                os.makedirs(tile_directory_hq)
            for y in range(math.floor(-2**depth), math.ceil(2**depth)):
                tile_addresses.append((x, y))

        random.shuffle(tile_addresses)

        for x, y in tile_addresses:
            pool.apply_async(try_create_tile, args=(depth, x, y), callback=on_complete)
        
        pool.close()
        pool.join()