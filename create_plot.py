import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
from config import *

codes = np.load(LATENT_CODES_EMBEDDED_FILE_NAME)

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from image_loader import ImageDataset
dataset = ImageDataset()

width, height = 80, 80


fig, ax = plt.subplots()
plt.axis('off')
margin = 0.0128
plt.margins(margin * height / width, margin)

x = codes[:, 0]
y = codes[:, 1]
x = np.interp(x, (x.min(), x.max()), (0, 1))
y = np.interp(y, (y.min(), y.max()), (0, 1))

ax.scatter(x, y, s = 40, cmap='Set1')
fig.set_size_inches(width, height)

for i in tqdm(range(codes.shape[0])):
    image = dataset[i].numpy().transpose((1, 2, 0))
    box = AnnotationBbox(OffsetImage(image, zoom = 0.25, cmap='gray'), (x[i], y[i]), frameon=False)
    ax.add_artist(box)
    
print("Saving plot...")

extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
plt.savefig('tsne.png', bbox_inches=extent, dpi=200)