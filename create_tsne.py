import numpy as np
import os
from MulticoreTSNE import MulticoreTSNE as TSNE

latent_codes = np.load('data/latent_codes.npy')

tsne = TSNE(n_jobs=os.cpu_count())
embedded = tsne.fit_transform(latent_codes)
np.save('data/latent_codes_embedded.npy', embedded)