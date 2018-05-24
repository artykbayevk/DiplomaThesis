import os
import glob
import numpy as np
import skimage
import shutil
import json


BASE_DATA_DIR = '../data/fl47'
className2ClassIDFile = os.path.join(BASE_DATA_DIR,'className2ClassID.txt')
initialTrainDir = os.path.join(BASE_DATA_DIR, 'train')
initialTestDir = os.path.join(BASE_DATA_DIR, 'test')

copied = True


###COPYING TRAIN AND TEST IMAGES TO ONE DIRECTORIES

if(copied == False):
    try:
        os.makedirs(os.path.join(BASE_DATA_DIR, 'train','original'))
        os.makedirs(os.path.join(BASE_DATA_DIR, 'test','original'))
    except OSError as exc:
        pass


    files_000000 = glob.glob(os.path.join(BASE_DATA_DIR, 'train','000000','*.png'))
    files_000001 = glob.glob(os.path.join(BASE_DATA_DIR, 'train','000001','*.png'))
    files_000002 = glob.glob(os.path.join(BASE_DATA_DIR, 'train','000002','*.png'))

    files_000003 = glob.glob(os.path.join(BASE_DATA_DIR, 'test','000000','*.png'))
    files_000004 = glob.glob(os.path.join(BASE_DATA_DIR, 'test','000001','*.png'))
    files_000005 = glob.glob(os.path.join(BASE_DATA_DIR, 'test','000002','*.png'))
    for item in files_000000:
        tmp = item.split('mask')
        if(len(tmp)==1):
            shutil.copy(item, os.path.join(BASE_DATA_DIR,'train','original'))
    for item in files_000001:
        tmp = item.split('mask')
        if(len(tmp)==1):
            shutil.copy(item, os.path.join(BASE_DATA_DIR,'train','original'))
    for item in files_000002:
        tmp = item.split('mask')
        if(len(tmp)==1):
            shutil.copy(item, os.path.join(BASE_DATA_DIR,'train','original'))
    for item in files_000003:
        tmp = item.split('mask')
        if(len(tmp)==1):
            shutil.copy(item, os.path.join(BASE_DATA_DIR,'test','original'))
    for item in files_000004:
        tmp = item.split('mask')

        if(len(tmp)==1):
            shutil.copy(item, os.path.join(BASE_DATA_DIR,'test','original'))
    for item in files_000005:
        tmp = item.split('mask')
        if(len(tmp)==1):
            shutil.copy(item, os.path.join(BASE_DATA_DIR,'test','original'))

fromClassFileData = open(className2ClassIDFile, 'rb')
classes = []


### READING CLASSES ID
for item in fromClassFileData:
    classes.append([item.split('\t')[0], item.split('\t')[1].split('\n')[0]])
# print(classes)


###READING ALL TXT FILES FROM TRAIN AND TEST DIRECTORIES
trainTxtFileDir00 = glob.glob(os.path.join(BASE_DATA_DIR, 'train','000000','*.txt'))
trainTxtFileDir01 = glob.glob(os.path.join(BASE_DATA_DIR, 'train','000001','*.txt'))
trainTxtFileDir02 = glob.glob(os.path.join(BASE_DATA_DIR, 'train','000002','*.txt'))

testTxtFileDir00 = glob.glob(os.path.join(BASE_DATA_DIR, 'test','000000','*.txt'))
testTxtFileDir01 = glob.glob(os.path.join(BASE_DATA_DIR, 'test','000001','*.txt'))
testTxtFileDir02 = glob.glob(os.path.join(BASE_DATA_DIR, 'test','000002','*.txt'))

trainTxtFiles = trainTxtFileDir00+trainTxtFileDir01+trainTxtFileDir02
testTxtFiles = testTxtFileDir00+testTxtFileDir01+testTxtFileDir02

# print(len(trainTxtFiles))
# print(len(testTxtFiles))

crops = {}
for item in trainTxtFiles:
    tmp_file = open(item,'rb')
    filename = item.split('/')[-1].split('.')[0]+'.txt'
    content = tmp_file.readlines()
    content = [x.split('\n')[0] for x in content]
    crops[filename] = content


with open('train.json', 'w') as fp:
    json.dump(crops, fp)
