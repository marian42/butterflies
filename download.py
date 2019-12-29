from tqdm import tqdm
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import os
import metadata

items = metadata.load()

for item in tqdm(items):
    file_name = item.image_filename
    
    if os.path.exists(file_name):
        continue

    try:
        request = Request(item.image_url)
        image_file = open(file_name, 'wb')
        image_file.write(urlopen(request).read())
        image_file.close()
    except HTTPError:
        print("Error downloading file {:s}".format(item.image_url))
