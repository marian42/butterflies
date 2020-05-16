import os
import numpy as np
from skimage import io, transform
from tqdm import tqdm
import webp

OUTPUT_RESOLUTION = 128

from rotation_dataset import RotationDataset
dataset = RotationDataset()

file_names = ['data/images_alpha/{:s}.webp'.format(id) for id in dataset.image_ids]

os.makedirs('data/images_alpha/', exist_ok=True)

for file_name in tqdm(file_names):    
    hash = file_name.split('/')[-1][:-5]
    out_file_name = 'data/images_{:d}/{:s}.jpg'.format(OUTPUT_RESOLUTION, hash)
    if os.path.exists(out_file_name):
        continue
    try:
        image = webp.imread(file_name).astype(np.float32)
        alpha_mask = image[:, :, 3][:, :, np.newaxis]
        image = image[:, :, :3] * alpha_mask / 255 + (255.0 - alpha_mask)

        image = transform.resize(image, (OUTPUT_RESOLUTION, OUTPUT_RESOLUTION), preserve_range=True).astype(np.uint8)
        io.imsave(out_file_name, image, quality=98)
    except Exception as exception:
        if isinstance(exception, KeyboardInterrupt) or True:
            raise exception
        print(("Error while handling {:s}".format(file_name)))