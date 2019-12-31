import os
import numpy as np
import glob
from skimage import io, transform
from tqdm import tqdm

OUTPUT_RESOLUTION = 128

file_names = glob.glob(os.path.join('data/images_alpha/', '**.png'), recursive=True)

for file_name in tqdm(file_names):    
    hash = file_name.split('/')[-1][:-4]
    out_file_name = 'data/images_{:d}/{:s}.jpg'.format(OUTPUT_RESOLUTION, hash)
    if os.path.exists(out_file_name):
        continue
    try:
        image = io.imread(file_name)
        image = image[:, :, :3]
        image = transform.resize(image, (OUTPUT_RESOLUTION, OUTPUT_RESOLUTION), preserve_range=True).astype(np.uint8)
        io.imsave(out_file_name, image)
    except Exception as exception:
        if isinstance(exception, KeyboardInterrupt):
            raise exception
        print(("Error while handling {:s}".format(file_name)))