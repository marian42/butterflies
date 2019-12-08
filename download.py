
import csv


from tqdm import tqdm
from urllib.request import Request, urlopen
import os

def get_ids():
    ids_file = open('data/ids.txt', 'r')
    lines = ids_file.readlines()
    ids_file.close()
    return set(int(line.strip()) for line in lines)


def get_urls(ids):
    file = open('data/multimedia.csv', 'r')
    for row in csv.reader(file):
        try:
            id = int(row[0])
            if id in ids and 'label' not in row[2]:
                yield row[5]
        except:
            pass

ids = get_ids()
urls = list(get_urls(ids))

for url in tqdm(urls):
    image_id = url.split('/')[6]
    file_name = 'data/raw/{:s}.jpg'.format(image_id)
    
    if os.path.exists(file_name):
        continue

    request = Request(url)
    image_file = open(file_name, 'wb')
    image_file.write(urlopen(request).read())
    image_file.close()
