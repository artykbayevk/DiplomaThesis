import os
import glob
import cv2
import numpy as np
import skimage
import shutil
import json
from skimage import io
from skimage.transform import rescale
import uuid


BASE_DATA_DIR = '../data/fl47'
className2ClassIDFile = os.path.join(BASE_DATA_DIR,'className2ClassID.txt')
initialTrainDir = os.path.join(BASE_DATA_DIR, 'train')
initialTestDir = os.path.join(BASE_DATA_DIR, 'test')
trainOriginalImageDir = os.path.join(initialTrainDir, 'original')
testOriginalImageDir = os.path.join(initialTestDir,'original')

trainSet47 = os.path.join("../annotations/trainset47.txt")
testSet47 = os.path.join("../annotations/testset47.txt")


trainImageDir = os.path.join(initialTrainDir,'images')
testImageDir = os.path.join(initialTestDir,'images')


copied = True
annotationCreated = True
imageCropped = True
noLogoImagesAppended = True
###COPYING TRAIN AND TEST IMAGES TO ONE DIRECTORIES

if(copied == False):
    try:
        os.makedirs(os.path.join(BASE_DATA_DIR, 'train','original'))
        os.makedirs(os.path.join(BASE_DATA_DIR, 'test','original'))
        os.makedirs(os.path.join(BASE_DATA_DIR, 'train','images'))
        os.makedirs(os.path.join(BASE_DATA_DIR, 'test','images'))
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
classes = [x[0] for x in classes]
print(classes)


###READING ALL TXT FILES FROM TRAIN AND TEST DIRECTORIES
trainTxtFileDir00 = glob.glob(os.path.join(BASE_DATA_DIR, 'train','000000','*.txt'))
trainTxtFileDir01 = glob.glob(os.path.join(BASE_DATA_DIR, 'train','000001','*.txt'))
trainTxtFileDir02 = glob.glob(os.path.join(BASE_DATA_DIR, 'train','000002','*.txt'))

testTxtFileDir00 = glob.glob(os.path.join(BASE_DATA_DIR, 'test','000000','*.txt'))
testTxtFileDir01 = glob.glob(os.path.join(BASE_DATA_DIR, 'test','000001','*.txt'))
testTxtFileDir02 = glob.glob(os.path.join(BASE_DATA_DIR, 'test','000002','*.txt'))

trainTxtFiles = trainTxtFileDir00+trainTxtFileDir01+trainTxtFileDir02
testTxtFiles = testTxtFileDir00+testTxtFileDir01+testTxtFileDir02


####CREATING ANNOTATION
trainAnnotation = os.path.join(BASE_DATA_DIR,'train_annotation.txt')
testAnnotation = os.path.join(BASE_DATA_DIR, 'test_annotation.txt')
trainCropAnnotation = os.path.join(BASE_DATA_DIR,'train_crop_annotation.txt')
testCropAnnotation = os.path.join(BASE_DATA_DIR, 'test_crop_annotation.txt')
noLogoCropAnnotation = os.path.join(BASE_DATA_DIR, 'nologo_annotation.txt')
if (annotationCreated==False):

    trainWriter = open(trainAnnotation, 'w')
    testWriter = open(testAnnotation,'w')
    for item in trainTxtFiles:
        tmp_file = open(item,'rb')
        filename = item.split('/')[-1].split('.')[0]+'.png'
        content = tmp_file.readlines()
        content = [x.split('\n')[0] for x in content]
        for logos in content:
            coordinates = logos.split(' ')[0:4]
            classOfLogo = logos.split(' ')[4]
            intoFile = filename+' '+' '.join(coordinates)+' '+classOfLogo+' \n'
            trainWriter.write(intoFile)

    trainWriter.close()
    for item in testTxtFiles:
        tmp_file = open(item,'rb')
        filename = item.split('/')[-1].split('.')[0]+'.png'
        content = tmp_file.readlines()
        content = [x.split('\n')[0] for x in content]
        for logos in content:
            coordinates = logos.split(' ')[0:4]
            classOfLogo = logos.split(' ')[4]
            intoFile = filename+' '+' '.join(coordinates)+' '+classOfLogo+' \n'
            testWriter.write(intoFile)
    testWriter.close()


####READING FROM ANNOTATION AND CROP LOGOS WITH THEIR COORDINATES
annotationsFromTrain = open(trainAnnotation,'rb')
annotationsFromTest = open(testAnnotation,'rb')

if (imageCropped == False):
    contentFromTrain = annotationsFromTrain.readlines()
    contentFromTest = annotationsFromTest.readlines()

    croppedAnnotationForTrain = open(trainCropAnnotation,'w')
    croppedAnnotationForTest = open(testCropAnnotation,'w')

    for lines in contentFromTrain:
        tmp = lines.split(' ')[:-1]
        filename = tmp[0]
        new_filename = uuid.uuid4().hex+'.jpg'
        coordinates = tmp[1:-1]
        coordinates = [int(x) for x in coordinates]
        x1 = coordinates[0]
        y1 = coordinates[1]
        x2 = coordinates[2]
        y2 = coordinates[3]
        classOfLogo =  int(tmp[-1])
        tmpImage = cv2.imread(os.path.join(trainOriginalImageDir, filename))
        try:
            cropped_image = tmpImage[y1:y2,x1:x2]
            a = cropped_image.shape[0]
            b = cropped_image.shape[1]

            diff = 0
            if(a<=b):
                diff = 225.0/a
            else:
                diff = 225.0/b

            resized_cropped_images = rescale(cropped_image, diff)
            io.imsave(os.path.join(trainImageDir, new_filename),resized_cropped_images)
            tmp = new_filename+' '+classes[classOfLogo] +' \n'
            croppedAnnotationForTrain.write(tmp)
        except TypeError:
            print('nothing')
    croppedAnnotationForTrain.close()



    for lines in contentFromTest:
        tmp = lines.split(' ')[:-1]
        filename = tmp[0]
        new_filename = uuid.uuid4().hex+'.jpg'
        coordinates = tmp[1:-1]
        coordinates = [int(x) for x in coordinates]
        x1 = coordinates[0]
        y1 = coordinates[1]
        x2 = coordinates[2]
        y2 = coordinates[3]
        classOfLogo =  int(tmp[-1])
        tmpImage = cv2.imread(os.path.join(testOriginalImageDir, filename))
        try:
            cropped_image = tmpImage[y1:y2,x1:x2]
            a = cropped_image.shape[0]
            b = cropped_image.shape[1]

            diff = 0
            if(a<=b):
                diff = 225.0/a
            else:
                diff = 225.0/b
            resized_cropped_images = rescale(cropped_image, diff)
            io.imsave(os.path.join(testImageDir, new_filename),resized_cropped_images)
            tmp = new_filename+' '+classes[classOfLogo] +' \n'
            croppedAnnotationForTest.write(tmp)
        except TypeError:
            print('nothing')
    croppedAnnotationForTest.close()



#####ADDING NO LOGO IMAGES TO TRAIN DIR
nologoImages = []

if (noLogoImagesAppended == False):
    noLogoImages = glob.glob(os.path.join(BASE_DATA_DIR, 'train','no-logo','*.png'))
    nologoWriter = open(noLogoCropAnnotation, 'w')
    for item in noLogoImages:
        f_name = item.split('/')[-1]
        #copy into train dir
        shutil.copy(src=item, dst=os.path.join(trainImageDir,f_name))
        tmp = f_name+ ' '+ 'nologo'+' \n'
        nologoWriter.write(tmp)
    nologoWriter.close()

os.system("cat "+trainCropAnnotation+" "+noLogoCropAnnotation+" > "+trainSet47)
shutil.copy(testCropAnnotation, testSet47)
