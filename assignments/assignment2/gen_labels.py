import os

outp_dir = 'data_augmented'

labels = ['class0', 'class1', 'class2', 'class3', 'class4']

tr_labels = open('trainLabels.csv', 'w')
print >> tr_labels, "image,level"

d_set = 'train'
if (1 == 1):    
    print 'Data set:', d_set
    for l in labels:
	k = 0
	print '  Label:', l
        files = os.listdir(os.path.join(outp_dir, d_set, l))
        for f in files:
            filename = f[:-5]
            k += 1
	    print '    File:', f, '(' + str(k), '/', str(len(files)) + ')'
            print >> tr_labels, filename+','+l[-1]
