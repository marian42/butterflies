import os
import numpy as np
import glob
from skimage import transform
from tqdm import tqdm
import random
from torch.utils.data import DataLoader, Dataset
import multiprocessing
from config import *
import webp
import csv
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

OUTPUT_FILENAME = 'data/exported/{:s}.jpg'

MIN_QUALITY = 2
MIN_RESOLUTION = 512

MARGIN = 0.05

SHADOW_BRIGHTNESS = 0.4
BLUR_RADIUS = 0.015

BACKGROUND_COLOR = (240, 240, 240)

shadow_color = (int)(255 * SHADOW_BRIGHTNESS)

rotation_file = open(ROTATIONS_CALCULATED_FILENAME, 'r')
reader = csv.reader(rotation_file)
rotations = {row[0]: float(row[1]) for row in reader}
rotation_file.close()

quality_file = open(QUALITY_CALCULATED_FILENAME, 'r')
reader = csv.reader(quality_file)
quality = {row[0]: int(row[1]) for row in reader}
quality_file.close()

def clip_alpha(image, add_margin = MARGIN):
    ALPHA_THRESHOLD = 0.05
    coords = ((image[:, :, 3] > ALPHA_THRESHOLD)).nonzero()
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)
    image = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1], :]
    new_size = np.max(bottom_right - top_left)
    new_size = new_size + (int)(new_size * add_margin)
    result = np.ones((new_size, new_size, 4), dtype=np.float32)
    result[:, :, 3] = 0
    x, y = (new_size - image.shape[0]) // 2, (new_size - image.shape[1]) // 2
    result[x:x+image.shape[0], y:y+image.shape[1], :] = image
    return result


def handle_image(file_name):
    file_hash = file_name.split('/')[-1].split('.')[0]
    out_file_name = OUTPUT_FILENAME.format(file_hash)

    if os.path.isfile(out_file_name):
        return

    if file_hash not in quality or quality[file_hash] < MIN_QUALITY:
        #print("Image {:s} is low quality.".format(file_hash))
        return

    rotation = rotations[file_hash] if file_hash in rotations else 0

    # Read image
    image = webp.imread(file_name).astype(np.float32) / 255

    if image.shape[0] < MIN_RESOLUTION or image.shape[1] < MIN_RESOLUTION:
        #print("Image {:s} is too small.".format(file_hash))
        return

    alpha_mask = image[:, :, 3][:, :, np.newaxis]

    # Rotate
    image = transform.rotate(image, rotation, resize=True, clip=True, mode='constant', cval=0)

    # Crop
    image = clip_alpha(image)

    size = image.shape[0]

    # Shadow

    image_uint8 = (image * 255).astype(np.uint8)

    shadow_mask = Image.fromarray(image_uint8[:, :, -1], mode="L")
    shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(radius=(int)(size * BLUR_RADIUS)))
    
    # Composite    
    result = Image.new("RGB", (size, size), BACKGROUND_COLOR)
    result.paste((shadow_color, shadow_color, shadow_color), (0, 0), mask=shadow_mask)

    image_pil = Image.fromarray(image_uint8, mode="RGBA")
    result.paste(image_pil, (0, 0, size, size), mask=image_pil)

    # Save
    result.save(out_file_name, quality=95)

if __name__ == '__main__':
    file_names = glob.glob('data/images_alpha/**.webp', recursive=True)

    worker_count = os.cpu_count() - 2
    print("Using {:d} processes.".format(worker_count))
    context = multiprocessing.get_context('spawn')
    pool = context.Pool(worker_count)

    progress = tqdm(total=len(file_names))

    def on_complete(*_):
        progress.update()

    random.shuffle(file_names)

    for file_name in file_names:        
        pool.apply_async(handle_image, args=(file_name,), callback=on_complete)

    pool.close()
    pool.join()