
import csv
file = open('data/multimedia.csv', 'r')
reader = csv.reader(file)
rows = [row for row in reader]
rows = rows[1:]

from tqdm import tqdm
from urllib.request import Request, urlopen

index = 0

for row in tqdm(rows):
    request = Request(row[5])
    image_file = open('data/raw/{:d}.jpg'.format(index), 'wb')
    image_file.write(urlopen(request).read())
    image_file.close()
    index += 1
