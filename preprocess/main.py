from __future__ import print_function, division
import os
import numpy as np
import uuid
import glob

from skimage import io, transform
from random import randint

import warnings
warnings.filterwarnings("ignore")

input_ann_path = '../data/fl27/annotation.txt'
input_data_path = '../data/fl27/images'

output_ann_path = '../data/fl27/crop_annotation.txt'
output_data_path = '../data/fl27/cropped'

resized_image_path = '../data/fl27/resized'
train_test_dir_path ='../annotations'


def read_from_annotation(path):
	file = open(path, "r")
	content = file.readlines()
	new = [x.split(" ")[:-1] for x in content]
	return new

def create_crop(in_ann_path, in_img_path, out_ann_path, out_img_path):
	annotations = read_from_annotation(in_ann_path)
	out_ann = open(out_ann_path, "w")
	for i,ann in zip(range(len(annotations)),annotations):
		img_name = ann[0]
		img_path = os.path.join(in_img_path, img_name)
		image = io.imread(img_path)
		positions = ann[-4:]
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

		cropped = image[y1:y2, x1:x2]
		cropped_name = str(uuid.uuid4().hex)+'.jpg'
		cropped_class = ann[1]
		try:
			io.imsave(os.path.join(out_img_path, cropped_name), cropped)
			note = "{} {} \n".format(cropped_name, cropped_class) 
			print(cropped_name, cropped_class)
			out_ann.write(note)
		except IndexError:
			print('No writing')
	out_ann.close()

def resize_images_in_dir(in_directory_path, out_directory_path):
	filenames = glob.glob(in_directory_path+'/*.jpg')
	for file in filenames:
		img = io.imread(file)
		resized_img = transform.resize(img,(224, 224))
		fname = file.split('/')[-1]
		io.imsave(os.path.join(out_directory_path, fname), resized_img)

def prepare_labels(ann_path):
	arr = read_from_annotation(ann_path)
	labels = []
	for item in (arr):
		l = item[1]
		if l not in labels:
			labels.append(l)
		else:
			continue
	return labels

def train_test_split(crop_ann_path, out_directory, test_size = 1000):
	train = read_from_annotation(crop_ann_path)
	test = []
	for i in range(test_size):
		randIndex = randint(0,len(train)-1)
		test.append(train[randIndex])
		del(train[randIndex])

	trainset = file(os.path.join(out_directory, 'trainset.txt'), "w")
	testset = file(os.path.join(out_directory, 'testset.txt'), "w")

	for item in train:
		tmp = "{} {} \n".format(item[0], item[1])
		trainset.write(tmp)
	trainset.close()

	for item in test:
		tmp = "{} {} \n".format(item[0], item[1])
		testset.write(tmp)
	testset.close()

def main():

	# For creating crop images of logos
	# create_crop(input_ann_path, input_data_path, output_ann_path, output_data_path)

	# For resizing crop images and lead they to one size
	# resize_images_in_dir(output_data_path,resized_image_path)

	# For creating train and test dataset annotations
	train_test_split(output_ann_path, train_test_dir_path)


if __name__ == '__main__':
	main()