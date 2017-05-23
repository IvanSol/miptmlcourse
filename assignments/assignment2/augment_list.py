import skimage.transform as transf
import skimage
import skimage.io as io
from settings import *
#import matplotlib.pyplot as plt
import os
import numpy as np

inp_dir = 'data_src'
outp_dir = 'data_augmented3'
delta = 80


def augment(element):
    d_set, l, f = element
    img0 = io.imread(os.path.join(inp_dir, d_set, l, f))
    print(os.path.join(inp_dir, d_set, l, f))
    if (d_set == 'train'):
        for j in range(3):
            img = transf.rotate(img0, np.random.randint(-30, 30))
            var = np.random.randint(0, 2)
            if (var > 0):
                img = skimage.util.random_noise(img, var=0.001 * var)
            img = img[delta:-delta, delta:-delta]
            img = transf.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            filename = f[:-5] + '_' + str(j)
            io.imsave(os.path.join(outp_dir, d_set, l, filename + '.jpeg'), skimage.img_as_float(img))
    else:
        io.imsave(os.path.join(outp_dir, d_set, l, f),
                  transf.resize(img0[delta:-delta, delta:-delta], (IMG_WIDTH, IMG_HEIGHT)))


import sys
f=open(sys.argv[1], 'r')

for line in f:
	augment(line[:-1].split(' '))
