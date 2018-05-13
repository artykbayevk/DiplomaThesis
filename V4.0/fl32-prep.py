from __future__ import print_function, division
from skimage import io, transform
from random import randint
import shutil

import os, uuid, glob, warnings
import numpy as np

warnings.filterwarnings("ignore")


ORIGINAL_DATA_DIR = '/fs/vulcan-scratch/kamalsdu/diploma_data/fl32/originals'
CROPPED_DATA_DIR = '/fs/vulcan-scratch/kamalsdu/diploma_data/fl32/images'
ORIGINAL_ANNOTATION = '/fs/vulcan-scratch/kamalsdu/diploma_data/fl32/annotation.txt'
CROPPED_ANNOTATION = '/fs/vulcan-scratch/kamalsdu/diploma_data/fl32/crop_annotation.txt'

TRAIN_SET = '../annotations/trainset32.txt'
TEST_SET = '../annotations/testset32.txt'

path = '/fs/vulcan-scratch/kamalsdu/diploma_data/fl32/classes/masks'
f1 = open(ORIGINAL_ANNOTATION,'w')
for root, dirs, files in os.walk(path):
    if(len(files)!=0):
        for item in files:
            f = open(os.path.join(root,item), 'r')
            content = f.readlines()[1:]
            content = [x.split('\r\n')[:-1] for x in content]
            content = [x[0].split(' ') for x in content]
            for line in content:
                a = item.split('.bboxes.txt')[0]
                b = root.split('/')[-1]
                c = line
                into = '{} {} {} \n'.format(a,b,' '.join(c))
                f1.write(into)
            f.close()
f1.close()
