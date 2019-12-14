import numpy as np
import json

codes = np.load('data/latent_codes_embedded.npy')

min = np.min(codes, axis=0)
max = np.max(codes, axis=0)
codes -= (max + min) / 2
codes /= np.max(codes, axis=0)

from image_loader import ImageDataset
dataset = ImageDataset()

result = []
for i in range(codes.shape[0]):
    result.append({'x': codes[i, 0].item(), 'y': codes[i, 1].item(), 'hash': dataset.hashes[i]})

json_string = json.dumps(result)
with open('data/tsne.json', 'w') as file:
    file.write(json_string)