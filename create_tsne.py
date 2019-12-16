import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE

latent_codes = np.load('data/latent_codes.npy')

tsne = TSNE(n_jobs=4)
embedded = tsne.fit_transform(latent_codes)
np.save('data/latent_codes_embedded.npy', embedded)