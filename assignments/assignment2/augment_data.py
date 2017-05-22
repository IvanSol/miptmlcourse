import skimage.transform as transf
import skimage
import skimage.io as io
import matplotlib.pyplot as plt
import os
import numpy as np

inp_dir = 'data'
outp_dir = 'data2'

labels =  {'train': ['class0', 'class1', 'class2', 'class3', 'class4'],
 'validation': ['class0', 'class1', 'class2', 'class3', 'class4'],
 'test': ['']}
sets = ['train']#, 'validation', 'test']
delta = 80

tr_labels = open('trainLabels.csv', 'w')
print >> tr_labels, "image,level"

d_set = 'train'
if not os.path.exists(outp_dir):
    os.makedirs(outp_dir)

for d_set in sets:
    if not os.path.exists(os.path.join(outp_dir, d_set)):
        os.makedirs(os.path.join(outp_dir, d_set))
    for l in labels[d_set]:
        if not os.path.exists(os.path.join(outp_dir, d_set, l)):
            os.makedirs(os.path.join(outp_dir, d_set, l))
        for f in os.listdir(os.path.join(inp_dir, d_set, l)):
            img0 = io.imread(os.path.join(inp_dir, d_set, l, f))
            for j in xrange(3):
                img = transf.rotate(img0, np.random.randint(-30, 30))
                var = np.random.randint(0, 5)
                if (var > 0):
                    img = skimage.util.random_noise(img, var=0.001 * var)
                img = img[delta:-delta, delta:-delta]
                filename = f[:-5] + '_' + str(j)
                io.imsave(os.path.join(outp_dir, d_set, l, filename + '.jpeg'), skimage.img_as_float(img))
                if (d_set == 'train'):
                    print >> tr_labels, filename+','+l[-1]