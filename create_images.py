import glob
from torchvision import utils
from torch.utils.data import DataLoader, Dataset
from shutil import copyfile
from tqdm import tqdm
import os
import cv2
from skimage import io
import multiprocessing
import traceback
import numpy as np
from skimage import io
import torch

SKIP_ITEM = 0

SIZE_BLOCK = 128

class RawImageDataset(Dataset):
    def __init__(self):
        file_names = file_names = glob.glob('data/raw/**.jpg', recursive=True)
        self.hashes = [f.split('/')[-1][:-4] for f in file_names]
        self.skip_existing_files = True
        
    def __len__(self):
        return len(self.hashes)

    def __getitem__(self, index):
        hash = self.hashes[index]
        image_file_name = 'data/raw/{:s}.jpg'.format(hash)
        result_file_name = 'data/images_alpha/{:s}.png'.format(hash)

        if self.skip_existing_files and os.path.exists(result_file_name):
            return SKIP_ITEM

        try:
            image = io.imread(image_file_name)
            image = image.transpose((2, 0, 1)).astype(np.float32) / 255

            input_width = image.shape[2]
            input_height = image.shape[1]

            width = SIZE_BLOCK * (input_width // SIZE_BLOCK + 1)
            height = SIZE_BLOCK * (input_height // SIZE_BLOCK + 1)

            result = np.ones((3, height, width), dtype=np.float32)
            result[:, :image.shape[1], :image.shape[2]] = image

            image = torch.from_numpy(result)
        except:
            print("Could not open {:s}.".format(image_file_name))
            return SKIP_ITEM
        
        return image, result_file_name, input_width, input_height

def remove_smaller_components(mask):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    if stats.shape[0] < 2:
        return
    max_label = np.argmax(stats[1:, 4]) + 1
    mask[labels != max_label] = 0

def save_image(image, mask, file_name):
    image = image.squeeze(0).numpy()

    mask = mask.squeeze(0).numpy()
    mask = mask > 0.5
    remove_smaller_components(mask)
    coords = np.stack(mask.nonzero())

    if coords.size == 0:
        print("Found nothing.")
        return

    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)
    
    mask = mask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    image = image[:, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

    if image.shape[1] < 10 or image.shape[2] < 10:
        print("Found nothing.")
        return

    image = image * mask + (1.0 - mask) * 1

    new_size = int(max(image.shape[1], image.shape[2]))    
    result = np.ones((4, new_size, new_size))
    result[3, :, :] = 0
    y, x = (new_size - image.shape[1]) // 2, (new_size - image.shape[2]) // 2
    result[:3, y:y+image.shape[1], x:x+image.shape[2]] = image
    result[3, y:y+image.shape[1], x:x+image.shape[2]] = mask

    io.imsave(file_name, (result.transpose((1, 2, 0)) * 255).astype(np.uint8))

if __name__ == '__main__':
    import torch
    from classifier import Classifier
    from torch.utils.data import DataLoader, Dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CLASSIFIER_FILENAME = 'trained_models/classifier.to'

    classifier = Classifier()
    classifier.cuda()
    classifier.load_state_dict(torch.load(CLASSIFIER_FILENAME))
    classifier.eval()

    dataset = RawImageDataset()
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)

    worker_count = os.cpu_count()
    print("Using {:d} processes.".format(worker_count))
    context = multiprocessing.get_context('spawn')
    pool = context.Pool(worker_count)

    progress = tqdm(total=len(dataset))

    def on_complete(*_):
        progress.update()

    for item in data_loader:
        if item == SKIP_ITEM:
            progress.update()
            continue

        image, result_file_name, width, height = item
        width, height = width[0].item(), height[0].item()

        try:
            with torch.no_grad():
                mask = classifier(image.to(device)).squeeze(0).squeeze(0).cpu()
            image = image[0, :, :height, :width]
            mask = mask[:height, :width]
            pool.apply_async(save_image, args=(image, mask, result_file_name[0]), callback=on_complete)
        except Exception as exception:
            if isinstance(exception, KeyboardInterrupt):
                raise exception
            print(("Error while handling {:s}".format(result_file_name[0])))
            traceback.print_exc()
    pool.close()
    pool.join()