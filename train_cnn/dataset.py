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
	def __init__(self, txt_file, root):
		self.txt_file = txt_file
		self.root = root

	def __len__(self):
		return self.txt_file.shape[0]

	def __getitem__(self, id):
		img_name = os.path.join(self.root, self.txt_file[id][0])
		img = io.imread(img_name)
		logo = int(self.txt_file[id][1])
		image = img.transpose((2, 0, 1))
		img = torch.FloatTensor(image)
		return img,logo


def show_image(image, label):
	plt.title(label)
	plt.imshow(image)
	plt.pause(2)  # pause a bit so that plots are updated

def show_images(data,count):
	for i, sample in zip(range(count), data):
		img = io.imread(os.path.join(images_path, sample[0]))
		label = labels[int(sample[1])]
		show_image(img, label)


def show_batch(sample_batched, labels):
	images_batch, logo_batch = sample_batched
	batch_size = len(images_batch)
	im_size = images_batch.size(2)

	text = ""

	for item in logo_batch:
		text+=" "+labels[item]

	grid = utils.make_grid(images_batch)
	plt.imshow(grid.numpy().transpose((1, 2, 0)))
	for i in range(batch_size):
		plt.text(1, 0.65,text)
	plt.title('Batch from dataloader')

def draw_batch(dataloader, labels):
	for i, sample in enumerate(dataloader, 0):
		images, logos = sample
		print(images.size(), logos)
		if i == 3:
			plt.figure()
			show_batch(sample, labels)
			plt.axis('off')
			plt.ioff()
			plt.show()
			break


def get_dataset():
	labels = tl.prepare_labels(main_ann_path)
	
	train_data = tl.prepare_num_dataset(main_ann_path, train_set_path)
	test_data = tl.prepare_num_dataset(main_ann_path, test_set_path)
	
	
	
	trainset = MyDataset(train_data, images_path)
	testset = MyDataset(test_data, images_path)


	
	#Drawing some images from dataloader
	dataLoader = DataLoader(trainset, batch_size=4,shuffle=True, num_workers=4)
	draw_batch(dataLoader, labels)
	print('Shapes: ',train_data.shape, test_data.shape)

if __name__ == '__main__':
	get_dataset()