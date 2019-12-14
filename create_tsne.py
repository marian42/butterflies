import numpy as np
from sklearn.manifold import TSNE

latent_codes = np.load('data/latent_codes.npy')
tsne = TSNE(n_components=2)
embedded = tsne.fit_transform(latent_codes)
np.save('data/latent_codes_embedded.npy', embedded)