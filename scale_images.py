import os
import numpy as np
import glob
from skimage import io, transform
from tqdm import tqdm
import math
import random
import torch

OUTPUT_RESOLUTION = 128
ROTATE = True

if ROTATE:
    from rotation_network import RotationNetwork
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NETWORK_FILENAME = 'trained_models/rotation.to'
    network = RotationNetwork()
    network.load_state_dict(torch.load(NETWORK_FILENAME))
    network.cuda()
    network.eval()

file_names = glob.glob('data/images_alpha/**.png', recursive=True)
random.shuffle(file_names)

@torch.no_grad()
def get_rotation_angle(image):
    image = transform.resize(image, (64, 64), preserve_range=True)
    image = torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float32, device=device).unsqueeze(0) / 255
    result = network(image).squeeze()
    return -math.degrees(math.atan2(result[0], result[1]))

WHITE_THRESHOLD = 0.95

def clip_image(image):
    coords = ((image[:, :, 0] < WHITE_THRESHOLD) | (image[:, :, 1] < WHITE_THRESHOLD) | (image[:, :, 2] < WHITE_THRESHOLD)).nonzero()
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)
    center = ((top_left + bottom_right) / 2).astype(int)
    half_size = np.max(bottom_right - top_left).item() // 2
    return image[center[0] - half_size:center[0] + half_size, center[1]-half_size:center[1]+half_size, :]

for file_name in tqdm(file_names):    
    hash = file_name.split('/')[-1][:-4]
    out_file_name = 'data/images{:s}_{:d}/{:s}.jpg'.format('_rotated' if ROTATE else '', OUTPUT_RESOLUTION, hash)
    if os.path.exists(out_file_name):
        continue
    try:
        image = io.imread(file_name)
        image = image[:, :, :3]

        if ROTATE:
            angle = get_rotation_angle(image)
            image = transform.rotate(image, angle, resize=True, clip=True, mode='constant', cval=1)
            image = clip_image(image) * 255

        image = transform.resize(image, (OUTPUT_RESOLUTION, OUTPUT_RESOLUTION), preserve_range=True).astype(np.uint8)
        io.imsave(out_file_name, image)
    except Exception as exception:
        if isinstance(exception, KeyboardInterrupt):
            raise exception
        print(("Error while handling {:s}".format(file_name)))