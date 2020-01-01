from tqdm import tqdm
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import os
import metadata
import time

items = metadata.load()
progress = tqdm(total=len(items))
total_bytes = 0

for item in items:
    progress.update()
    file_name = item.image_filename
    
    if os.path.exists(file_name):
        continue

    try:
        start = time.time()
        request = Request(item.image_url)
        image_file = open(file_name, 'wb')
        image_file.write(urlopen(request).read())
        current_bytes = image_file.tell()
        image_file.close()
        total_bytes += current_bytes
        if total_bytes < 0.9e9:
            progress.desc = '{:0.1f} MB, {:.1f} MB/s'.format(total_bytes * 1e-6, current_bytes * 1e-6 / (time.time() - start))
        else:
            progress.desc = '{:0.1f} GB, {:.1f} MB/s'.format(total_bytes * 1e-9, current_bytes * 1e-6 / (time.time() - start))
    except HTTPError:
        print("Error downloading file {:s}".format(item.image_url))
