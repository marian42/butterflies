import numpy as np
from config import *

METHOD = 2

latent_codes = np.load('data/latent_codes.npy')

if METHOD == 1: 
    # multicore tsne
    # https://github.com/DmitryUlyanov/Multicore-TSNE
    from MulticoreTSNE import MulticoreTSNE as TSNE
    import os
    tsne = TSNE(n_jobs=os.cpu_count())
    embedded = tsne.fit_transform(latent_codes)
elif METHOD == 2:
    # opentsne, uses a single core
    # https://github.com/pavlin-policar/openTSNE
    from openTSNE import TSNE
    embedded = TSNE().fit(latent_codes)

np.save(LATENT_CODES_EMBEDDED_FILE_NAME, embedded)