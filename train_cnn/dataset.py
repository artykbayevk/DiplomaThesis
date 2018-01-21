from __future__ import print_function, division
import os
import torch
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tool as tl

from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

warnings.filterwarnings("ignore")

plt.ion()

main_ann_path = '../data/fl27/crop_annotation.txt'
images_path = '../data/fl27/resized'
train_set_path = '../annotations/trainset.txt'
test_set_path = '../annotations/testset.txt'


class MyDataset(Dataset):
	def __init__()

def show_image(image, label):
	plt.title(label)
	plt.imshow(image)
	plt.pause(2)  # pause a bit so that plots are updated

def show_images(data,count):
	for i, sample in zip(range(count), data):
		img = io.imread(os.path.join(images_path, sample[0]))
		label = labels[int(sample[1])]
		show_image(img, label)

def get_dataset():
	labels = tl.prepare_labels(main_ann_path)
	train_data = tl.prepare_num_dataset(main_ann_path, train_set_path)
	test_data = tl.prepare_num_dataset(main_ann_path, test_set_path)
	print('Shapes: ',train_data.shape, test_data.shape)

if __name__ == '__main__':
	get_dataset()