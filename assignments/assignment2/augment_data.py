import skimage.transform as transf
import skimage
import skimage.io as io
from settings import *
#import matplotlib.pyplot as plt
import os
import sys
import numpy as np

inp_dir = 'data_src'
outp_dir = 'data_augmented3'

labels =  {'train': ['class0', 'class1', 'class2', 'class3', 'class4'],
 'validation': ['class0', 'class1', 'class2', 'class3', 'class4'],
 'test': ['']}
sets = [
'train',
'validation',
'test'
]
delta = 80

CPU = int(sys.argv[1])
#tr_labels = open('trainLabels2.csv', 'w')

#print >> tr_labels, "image,level"

#d_set = 'train'
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

print('Total:', len(img_pool), 'images')

fs = []
for i in range(CPU):
    fs += [open('tmp/' + str(i), 'w')]

k = 0
for e in img_pool:
    print >> fs[k], e[0], e[1], e[2]
    #print(e[0], e[1], e[2], file=fs[k])
    k = (k + 1) % CPU

print('Done')