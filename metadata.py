import csv
from tqdm import tqdm

class Butterfly():
    def __init__(self, row):
        self.family = row[0]
        self.genus = row[1]
        self.species = row[2]
        self.subspecies = row[3]
        self.sex = row[4]
        self.latitude = row[5]
        self.longitude = row[6]
        self.country = row[7]
        self.name = row[8]
        self.name_author = row[9]
        self.day = row[10]
        self.month = row[11]
        self.year = row[12]
        self.id = int(row[13])
        self.occurence_id = row[14]
        self.image_id = row[15]

    @property
    def image_filename(self):
        return 'data/raw/{:s}.jpg'.format(self.image_id)

    @property
    def image_url(self):
        return 'https://www.nhm.ac.uk/services/media-store/asset/{:s}/contents/preview'.format(self.image_id)


def load():
    file = open('data/metadata.csv', 'r')
    reader = csv.reader(file)
    reader_iterator = iter(reader)
    column_names = next(reader_iterator)

    result = []

    row_index = 0
    progress = tqdm(total=727248, desc='Reading metadata.csv')

    for row in reader_iterator:
        result.append(Butterfly(row))
        progress.update()

    file.close()
    return result