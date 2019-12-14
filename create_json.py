import numpy as np
import json
from sklearn.cluster import KMeans
from itertools import count
from tqdm import tqdm

ITEMS_PER_QUAD = 16

codes = np.load('data/latent_codes_embedded.npy')

min_value = np.min(codes, axis=0)
max_value = np.max(codes, axis=0)
codes -= (max_value + min_value) / 2
codes /= np.max(codes, axis=0)

from image_loader import ImageDataset
dataset = ImageDataset()

def create_json_item(index):
    return {'x': codes[index, 0].item(), 'y': codes[index, 1].item(), 'hash': dataset.hashes[index]}

def select_indices(x_range, y_range):
    candidates = remaining_indices \
        & (codes[:, 0] > x_range[0]) \
        & (codes[:, 0] <= x_range[1]) \
        & (codes[:, 1] > y_range[0]) \
        & (codes[:, 1] <= y_range[1]) \

    candidate_codes = codes[candidates, :]

    if candidate_codes.shape[0] <= ITEMS_PER_QUAD:
        return candidates.nonzero()[0]

    kmeans = KMeans(n_clusters=ITEMS_PER_QUAD)
    indices = np.zeros(ITEMS_PER_QUAD, dtype=int)
    kmeans_clusters = kmeans.fit_predict(candidate_codes)
    for i in range(ITEMS_PER_QUAD):
        center = kmeans.cluster_centers_[i, :]
        dist = np.linalg.norm(candidate_codes - center[np.newaxis, :], axis=1)
        indices[i] = np.argmin(dist)
    return candidates.nonzero()[0][indices]

result = {}
remaining_indices = np.ones(codes.shape[0], dtype=bool)

for depth in count():
    quads = {}
    for x in tqdm(range(2**depth)):
        for y in range(2**depth):
            x_range = (-1 + x * 2 / (2**depth), -1 + (x+1) * 2 / (2**depth))
            y_range = (-1 + y * 2 / (2**depth), -1 + (y+1) * 2 / (2**depth))

            indices = select_indices(x_range, y_range)

            if len(indices) > 0:
                remaining_indices[indices] = 0
                quad = [create_json_item(i) for i in indices]
                if x not in quads:
                    quads[x] = {}
                quads[x][y] = {'items': quad}
    print("Depth {:d}, {:d} quads, {:d} remaining".format(depth, len(quads), np.count_nonzero(remaining_indices)))
    if len(quads) == 0 or np.count_nonzero(remaining_indices) < 100:
        break
    result[depth] = quads

json_string = json.dumps(result)
with open('data/tsne.json', 'w') as file:
    file.write(json_string)