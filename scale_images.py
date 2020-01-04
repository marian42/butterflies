import os
import numpy as np
import glob
from skimage import io, transform
from tqdm import tqdm
import math
import random
import torch
import multiprocessing

OUTPUT_RESOLUTION = 128
ROTATE = True

@torch.no_grad()
def get_rotation_angle(image):
    image = transform.resize(image, (64, 64), preserve_range=True)
    image = torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float32, device=device).unsqueeze(0) / 255
    result = network(image).squeeze()
    return -math.degrees(math.atan2(result[0], result[1]))

def clip_image(image):
    WHITE_THRESHOLD = 0.95
    coords = ((image[:, :, 0] < WHITE_THRESHOLD) | (image[:, :, 1] < WHITE_THRESHOLD) | (image[:, :, 2] < WHITE_THRESHOLD)).nonzero()
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)
    image = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1], :]
    new_size = np.max(bottom_right - top_left)
    result = np.ones((new_size, new_size, 3), dtype=np.float32)
    x, y = (new_size - image.shape[0]) // 2, (new_size - image.shape[1]) // 2
    result[x:x+image.shape[0], y:y+image.shape[1], :] = image
    return result

def get_result_file_name(hash):
    return 'data/images{:s}_{:d}/{:s}.jpg'.format('_rotated' if ROTATE else '', OUTPUT_RESOLUTION, hash)

def handle_image(hash, angle):
    try:
        image = io.imread('data/images_alpha/{:s}.png'.format(hash))
        image = image[:, :, :3]

        if angle is not None:
            image = transform.rotate(image, angle, resize=True, clip=True, mode='constant', cval=1)
            image = clip_image(image) * 255

        image = transform.resize(image, (OUTPUT_RESOLUTION, OUTPUT_RESOLUTION), preserve_range=True).astype(np.uint8)
        io.imsave(get_result_file_name(hash), image)
    except:
        print("Error while handling {:s}".format(hash))
        traceback.print_exc()

if __name__ == '__main__':
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

    worker_count = os.cpu_count()
    print("Using {:d} processes.".format(worker_count))
    context = multiprocessing.get_context('spawn')
    pool = context.Pool(worker_count)

    progress = tqdm(total=len(file_names))

    def on_complete(*_):
        progress.update()

    for file_name in file_names:
        hash = file_name.split('/')[-1][:-4]
        
        if os.path.exists(get_result_file_name(hash)):
            progress.update()
            continue

        if ROTATE:
            try:
                image = io.imread(file_name)
                angle = get_rotation_angle(image[:, :, :3])
            except KeyboardInterrupt:
                break
            except:
                print(("Error while loading {:s}".format(file_name)))
                progress.update()
                continue
        else:
            angle = None

        pool.apply_async(handle_image, args=(hash, angle), callback=on_complete)

    pool.close()
    pool.join()