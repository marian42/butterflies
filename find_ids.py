from tqdm import tqdm
import csv
file = open('data/occurrence.csv', 'r')
reader = csv.reader(file)

ids_filename = open('data/ids.txt', 'w')

for row in tqdm(reader):
    id = row[0]
    higher_classification = row[25]
    if 'papilionoidea' in higher_classification.lower():
        ids_filename.write(id + '\n')

ids_filename.close()
