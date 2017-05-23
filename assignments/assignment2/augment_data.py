import skimage.transform as transf
import skimage
import skimage.io as io
from settings import *
#import matplotlib.pyplot as plt
import os
import numpy as np

inp_dir = 'data'
outp_dir = 'data_augmented2'

labels =  {'train': ['class0', 'class1', 'class2', 'class3', 'class4'],
 'validation': ['class0', 'class1', 'class2', 'class3', 'class4'],
 'test': ['']}
sets = [
'train',
'validation',
'test'
]
delta = 80

import multiprocessing
CPUs = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=CPUs)

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
                  transf.resize(img0[delta:-delta][delta:-delta], (IMG_WIDTH, IMG_HEIGHT)))


tr_labels = open('trainLabels2.csv', 'w')

#print >> tr_labels, "image,level"

d_set = 'train'
if not os.path.exists(outp_dir):
    os.makedirs(outp_dir)

img_pool = []
for d_set in sets:
    #print 'Data set:', d_set
    if not os.path.exists(os.path.join(outp_dir, d_set)):
        os.makedirs(os.path.join(outp_dir, d_set))
    for l in labels[d_set]:
        k = 0
        #print '  Label:', l
        if not os.path.exists(os.path.join(outp_dir, d_set, l)):
            os.makedirs(os.path.join(outp_dir, d_set, l))
        files = os.listdir(os.path.join(inp_dir, d_set, l))
        for f in files:
            #print '    File:', f, '(' + str(k), '/', str(len(files)) + ')'
            k += 1
            img_pool += [[d_set, l, f]]

print('Running through', CPUs, 'processes...')
pool.map(augment, img_pool)
print('Done')