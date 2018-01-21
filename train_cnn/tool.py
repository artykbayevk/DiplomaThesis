from __future__ import print_function, division
import os
import numpy as np
import uuid
import glob

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

def prepare_num_dataset(annotation_path, set_path):
	labels = prepare_labels(annotation_path)
	arr = read_from_annotation(set_path)
	out = []
	for item in arr:
		tmp = [item[0], labels.index(item[1])]
		out.append(tmp)
	out = np.array(out)
	return out