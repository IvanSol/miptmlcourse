from models.settings import *
import numpy as np
import pandas as pd
import os

val_augmentation_rate = 4
train_augmentation_rate = 4
noise_pattern = [0.03] * NB_FEATURES
noise_pattern = np.reshape(np.tile(noise_pattern, NB_TIMESTEPS), (NB_TIMESTEPS, NB_FEATURES))

def noise_me(filename, i):
    ts = pd.read_csv(filename, index_col=0)
    shape = ts.as_matrix().shape
    rnorm = np.random.normal(loc = 0, scale = noise_pattern)
    out_filename = filename[:-4] + '_noised_' + str(i) + '.csv'
    (ts + rnorm).to_csv(out_filename, sep = ',')

input_dir = VALIDATION_DATA_DIR
for activity in ACTIVITIES_LIST:
    for _, _, filenames in os.walk(os.path.join(input_dir, activity)):
        for filename in filenames:
            for i in range(val_augmentation_rate):
                cur_filename = os.path.join(input_dir, activity, filename)
                noise_me(cur_filename, i)

input_dir = TRAIN_DATA_DIR
for activity in ACTIVITIES_LIST:
    for _, _, filenames in os.walk(os.path.join(input_dir, activity)):
        for filename in filenames:
            for i in range(train_augmentation_rate):
                cur_filename = os.path.join(input_dir, activity, filename)
                noise_me(cur_filename, i)