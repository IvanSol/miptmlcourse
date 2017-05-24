import os
from random import sample
from os.path import join
from models.settings import *

validation_rate = 0.2
labels = os.listdir(TRAIN_DATA_DIR)

for l in labels:
    files = os.listdir(join(VALIDATION_DATA_DIR, l))
    for f in files:
        if (not os.path.isfile(join(TRAIN_DATA_DIR, l, f))):
            os.rename(join(VALIDATION_DATA_DIR, l, f), join(TRAIN_DATA_DIR, l, f))
        else:
            os.remove(join(VALIDATION_DATA_DIR, l, f))

print("New validation rate is", validation_rate)

for l in labels:
    files = os.listdir(TRAIN_DATA_DIR + l)
    L = len(files)
    val_files_num = int(validation_rate * L)
    val_files = sample(files, val_files_num)
    for val_file in val_files:
        os.rename(join(TRAIN_DATA_DIR, l, val_file), join(VALIDATION_DATA_DIR, l, val_file))