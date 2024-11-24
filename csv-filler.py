from csv import writer
from os import listdir

allFiles = {
    0: 'test/neg',
    0: 'train/neg',
    1: 'test/pos',
    1: 'train/pos',
}

with open('data.csv', 'a') as file:
    w = writer(file)
    for value, path in allFiles.items():        
        for f in listdir(path):
            words = open(path + "/" + f, encoding="utf-8")
            review = words.read()
            try:
                w.writerow([value, review, (f.split('.')[0]).split('_')[-1]])
            except UnicodeEncodeError:
                continue