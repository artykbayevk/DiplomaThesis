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
train_set_path = '../annotations/trainset.txt'
test_set_path = '../annotations/testset.txt'




print (type(tl.prepare_num_dataset(main_ann_path, train_set_path)))
print (type(tl.prepare_num_dataset(main_ann_path, test_set_path)))