
import csv
file = open('data/multimedia.csv', 'r')
reader = csv.reader(file)
rows = [row for row in reader]
rows = rows[1:]

from tqdm import tqdm
from urllib.request import Request, urlopen

for row in tqdm(rows):
    request = Request(row[5])
    image_file = open('data/raw/{:s}.jpg'.format(row[0]), 'wb')
    image_file.write(urlopen(request).read())
    image_file.close()
