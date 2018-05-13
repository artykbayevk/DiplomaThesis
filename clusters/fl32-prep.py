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
# f1 = open(ORIGINAL_ANNOTATION,'w')
# for root, dirs, files in os.walk(path):
#     print(dirs)
#     if(len(files)!=0):
#         for item in files:
#             f = open(os.path.join(root,item), 'r')
#             content = f.readlines()[1:]
#             content = [x.split(' ') for x in content]
#             print(content)
#             for line in content:
#                 a = item.split('.bboxes.txt')[0]
#                 b = root.split('/')[-1]
#                 c = ((' '.join(line)).split('\r\n')[0])
#                 into = '{} {} {} \n'.format(a,b,c)
#                 print(into)
#                 f1.write(into)
#             f.close()
# f1.close()

def read_from_annotation(path):
    file = open(path, "r")
    content = file.readlines()
    new = [x.split(" ") for x in content]
    return new


# x = read_from_annotation(ORIGINAL_ANNOTATION)
# print(len(x))
def prepare_labels(annotaton_path):
    arr = read_from_annotation(annotaton_path)
    labels = []
    for item in (arr):
        l = item[1]
        if l not in labels:
            labels.append(l)
        else:
            continue
    return labels
LABELS = prepare_labels(ORIGINAL_ANNOTATION)
print(LABELS)
print(len(LABELS))


def create_resized_crops(input_annotation, input_directory,
                        output_annotation, output_directory):
    annotation = read_from_annotation(input_annotation)
    output_annotation = open(output_annotation, 'w')
    for i, data in zip(range(len(annotation)),annotation):
        tmp = ((' '.join(data)).split('\n')[0]).split(' ')
        data = tmp
        img_name = data[0]
        img_path = os.path.join(input_directory, img_name)
        image = io.imread(img_path)
        print(data)
        positions = data[2:6]
        print(positions)
        x1 = int(positions[0])
        y1 = int(positions[1])
        x2 = int(positions[2])
        y2 = int(positions[3])

        if x1 > x2:
            tmp = x1
            x1 = x2
            x2 = tmp

        if y1 > y2:
            tmp = y1
            y1 = y2
            y2 = tmp



        target = data[1]
#         fl32 - dataset
        crop_resizer(image, target, x1,x1+x2,y1,y1+y2,output_annotation, output_directory)
#
#         fl27 - dataset
#         crop_resizer(image, target, x1,x2,y1,y2,output_annotation, output_directory)
#         crop_resizer(image, target, x1-40,x2,y1,y2,output_annotation, output_directory)
#         crop_resizer(image, target, x1,x2+40,y1,y2,output_annotation, output_directory)
#         crop_resizer(image, target, x1,x2,y1-40,y2,output_annotation, output_directory)
#         crop_resizer(image, target, x1,x2,y1,y2+40,output_annotation, output_directory)

def crop_resizer(image,target, x1,x2,y1,y2, writer,directory):
    crop_img = image[y1:y2, x1:x2]
    w = crop_img.shape[0]
    h = crop_img.shape[1]
    file_name = str(uuid.uuid4().hex)+'.jpg'
    try:
        try:
            if(w<224 or h<224):
                scale = (224.0/w)+((224.0/w)*0.1) if w<h  else (224.0/h)+((224.0/h)*0.1)
                scaled = transform.rescale(crop_img, scale)
#                 print(scale)
#                 print(crop_img.shape)
#                 print(scaled.shape)
#                 print('--')
                io.imsave(os.path.join(directory, file_name), scaled)
                note = "{} {} \n".format(file_name, target)
                writer.write(note)
            else:
                io.imsave(os.path.join(directory, file_name), crop_img)
                note = "{} {} \n".format(file_name, target)
                writer.write(note)
        except ValueError:
            pass
    except IndexError:
        pass
# create_resized_crops(ORIGINAL_ANNOTATION, ORIGINAL_DATA_DIR, CROPPED_ANNOTATION, CROPPED_DATA_DIR)
NO_LOGO_DIRECTORY = '/fs/vulcan-scratch/kamalsdu/diploma_data/fl32/classes/no-logo'
NO_LOGO_DATA = glob.glob(os.path.join(NO_LOGO_DIRECTORY,'*.jpg'))
len(NO_LOGO_DATA)

images = []
# for item in NO_LOGO_DATA:
#     img_name = item.split('/')[-1]
#     im = io.imread(item)
#     io.imsave( os.path.join(CROPPED_DATA_DIR,img_name), im)
#     images.append([img_name, 'nologo'])
# print(len(images))
no_logo_annotation = '../data/fl32/nologo.txt'


import random
def train_test_split(crop_ann_path,  test_size = 1000):
    train = read_from_annotation(crop_ann_path)
    test = []
    for i in range(test_size):
        randIndex = randint(0,len(train)-1)
        test.append(train[randIndex])
        del(train[randIndex])

    print(len(train))
    print(len(test))
    random.shuffle(train)


    trainset = open(TRAIN_SET, "w")
    testset = open(TEST_SET, "w")

    for item in train:
        tmp = "{} {} \n".format(item[0], item[1])
        trainset.write(tmp)
    trainset.close()

    for item in test:
        tmp = "{} {} \n".format(item[0], item[1])
        testset.write(tmp)
    testset.close()
train_test_split(CROPPED_ANNOTATION,2500)
