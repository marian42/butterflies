from tkinter import *
import numpy as np
import json
from sklearn.cluster import KMeans
from itertools import count
from tqdm import tqdm
from PIL import Image, ImageDraw
import os

codes = np.load('data/latent_codes_embedded.npy')
TILE_FILE_FORMAT = 'data/tiles/{:d}/{:d}/{:d}.jpg'

DEPTH_OFFSET = 8

TILE_SIZE = 256
IMAGE_SIZE = 128
TILE_DEPTH = 7

min_value = np.min(codes, axis=0)
max_value = np.max(codes, axis=0)
codes -= (max_value + min_value) / 2
codes /= np.max(codes, axis=0)

from image_loader import ImageDataset
dataset = ImageDataset()

for x in tqdm(range(-2**TILE_DEPTH, 2**TILE_DEPTH), desc="Rendering"):
    for y in range(-2**TILE_DEPTH, 2**TILE_DEPTH):
        tile_file_name = TILE_FILE_FORMAT.format(TILE_DEPTH + DEPTH_OFFSET, x, y)
        if os.path.exists(tile_file_name):
            continue

        x_range = ((x - 0.5) / 2**TILE_DEPTH, (x + 1.5) / 2**TILE_DEPTH)
        y_range = ((y - 0.5) / 2**TILE_DEPTH, (y + 1.5) / 2**TILE_DEPTH)

        mask = (codes[:, 0] > x_range[0]) \
            & (codes[:, 0] < x_range[1]) \
            & (codes[:, 1] > y_range[0]) \
            & (codes[:, 1] < y_range[1])
        indices = mask.nonzero()[0]

        positions = codes[indices, :]
        positions *= 2**TILE_DEPTH * TILE_SIZE
        positions -= np.array((x * TILE_SIZE, y * TILE_SIZE))[np.newaxis, :]

        tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (255, 255, 255))
        
        for i in range(indices.shape[0]):
            index = indices[i]
            image_file_name = 'data/images_128/{:s}.jpg'.format(dataset.hashes[index])
            image = Image.open(image_file_name)
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
            tile.paste(image, (int(positions[i, 0] - IMAGE_SIZE // 2), int(positions[i, 1] - IMAGE_SIZE // 2)))
        
        tile_directory = os.path.dirname(tile_file_name)
        if not os.path.exists(tile_directory):
            os.makedirs(tile_directory)
        tile.save(tile_file_name)

for depth in range(TILE_DEPTH - 1, -1, -1):
    for x in tqdm(range(-2**depth, 2**depth), desc='Depth {:d}'.format(depth)):
        for y in range(-2**depth, 2**depth):
            tile_file_name = TILE_FILE_FORMAT.format(depth + DEPTH_OFFSET, x, y)
            if os.path.exists(tile_file_name):
                continue
            
            tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (255, 255, 255))
        
            for a in range(2):
                for b in range(2):
                    old_tile_file_name = TILE_FILE_FORMAT.format(depth + 1 + DEPTH_OFFSET, x * 2 + a, y * 2 + b)
                    image = Image.open(old_tile_file_name)
                    image = image.resize((TILE_SIZE // 2, TILE_SIZE // 2))
                    tile.paste(image, (a * TILE_SIZE // 2, b * TILE_SIZE // 2))
            
            tile_directory = os.path.dirname(tile_file_name)
            if not os.path.exists(tile_directory):
                os.makedirs(tile_directory)
            tile.save(tile_file_name)            
