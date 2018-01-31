from __future__ import print_function, division
import os
import numpy as np
import uuid
import glob
import time

from skimage import io, transform
from random import randint

import warnings
warnings.filterwarnings("ignore")

def read_from_annotation(path):
	file = open(path, "r")
	content = file.readlines()
	new = [x.split(" ")[:-1] for x in content]
	return new

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

def prepare_num_dataset(annotation_path, set_path, no_labels = False):
	if no_labels:
		labels = [-1]
	else:
		labels = prepare_labels(annotation_path)
	arr = read_from_annotation(set_path)
	out = []
	for item in arr:
		tmp = [item[0], labels.index(item[1])]
		out.append(tmp)
	out = np.array(out)
	return out

def resize_test_img(in_path, out_path):
	files = glob.glob(os.path.join(in_path, '*.jpg'))
	for f in files:
		img = io.imread(f)
		res = transform.resize(img, (224, 224))
		new_res_img_name = f.split('/')[-1]
		new_path = os.path.join(out_path, new_res_img_name)
		io.imsave(new_path, res)

def prepare_annotation(ann_path, img_path):
	all_files = glob.glob(os.path.join(img_path, '*.jpg'))
	f_writer = open(ann_path, 'w')
	for f in all_files:
		tmp = "{} {} \n".format(f.split('/')[-1], -1)
		f_writer.write(tmp)
	f_writer.close()

def prepare_test_img(path):
	out_dir = '../data/fl27/segmented_resized/not_predicted'
	whole_folders = glob.glob(os.path.join(path,'*'))
	for item in whole_folders:
		resize_test_img(item, out_dir)
	ann_path = '../annotations/segmentset.txt'
	prepare_annotation(ann_path, out_dir)
