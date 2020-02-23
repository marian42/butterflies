import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KDTree
from config import *
from multiprocessing import Pool
import os
import traceback
import time
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from matplotlib.collections import EllipseCollection

RADIUS = IMAGE_SIZE / 2 / TILE_SIZE / 2**TILE_DEPTH

plt.figure(num=None, figsize=(14, 14), dpi=140, facecolor='w', edgecolor='k')  

def plot(points, index):
    axes = plt.gca()
    x = points[:, 0]
    y = points[:, 1]
    axes.set_aspect('equal', adjustable='box')

    margin = 0.005
    
    plt.xlim((x_range[0] - margin, x_range[1] + margin))
    plt.ylim((y_range[0] - margin, y_range[1] + margin))

    distances, indices = KDTree(points).query(points, k=2, return_distance=True)
    violating_points = distances[:, 1] < 2 * RADIUS * 0.99

    size = RADIUS * 2
    
    axes.add_collection(EllipseCollection(widths=size, heights=size, angles=0, units='xy', offsets=points[~violating_points, :], transOffset=axes.transData, facecolor='None', edgecolor='gray'))
    axes.add_collection(EllipseCollection(widths=size, heights=size, angles=0, units='xy', offsets=points[violating_points, :], transOffset=axes.transData, facecolor='None', edgecolor='red'))
    
    plt.savefig('images/{:05d}.png'.format(index), bbox_inches='tight')
    plt.clf()


points = np.load('data/latent_codes_embedded.npy')
min_value = np.min(points, axis=0)
max_value = np.max(points, axis=0)
points -= (max_value + min_value) / 2
points /= np.max(points, axis=0)

#x_range = (-0.54, -0.44)
#y_range = (0.14, 0.225)
#mask = (points[:, 0] > x_range[0]) & (points[:, 0] < x_range[1]) & (points[:, 1] > y_range[0]) & (points[:, 1] < y_range[1])
#points = points[mask, :]

VELOCITY_DECAY = 0.8
REPELL_FORCE_STRENGTH = 0.5
RESET_FORCE_STRENGTH = 0.2 * RADIUS

def move_points(points, max_iter=400):
    velocity = np.zeros(points.shape)
    original_points = np.array(points)

    is_finalizing_phase = False
    finalizing_steps_left = 40

    for step in range(max_iter):
        #plot(points, step)

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
            if np.count_nonzero(distances[:, 1] < 2 * RADIUS * 0.85) < points.shape[0] * 0.01:
                is_finalizing_phase = True

        if is_finalizing_phase:
            finalizing_steps_left -= 1
            if finalizing_steps_left == 0:
                break


move_points(points)