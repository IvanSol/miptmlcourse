import os
from models.settings import *
from models.model import initialize_model
import numpy as np
import pandas as pd
import sys

# initialize model
model = initialize_model()
model.load_weights(sys.argv[1])

def read(file):
    data = pd.read_csv(file, index_col=0)
    return np.array(data.values.tolist())

input_files = os.listdir(TEST_DATA_DIR)

X = np.array(list(map(lambda x: read(os.path.join(TEST_DATA_DIR, x)), input_files)))
y = model.predict(X, batch_size=32).argmax(axis=1)

f = open('result.csv', 'w')
if PYTHON2:
    print >> f, 'file,activity'
else:
    print('file,activity', file = f)
# for i in range(min(y, 10)):
for i in range(len(y)):
    if PYTHON2:
        print >> f, str(input_files[i][:-4]) + ',' + str(y[i] + 1)
    else:
        print(input_files[i][:-4], y[i] + 1, file = f)
f.close()