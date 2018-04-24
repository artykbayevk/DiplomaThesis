import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from torch import optim

from skimage import io, transform
from random import randint
import shutil

import os
import uuid
import glob
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import time
import alexnet
from PIL import Image


warnings.filterwarnings("ignore")

plt.ion()

# 1 or 0
DATASET = 1

# 1 or hz
MODEL = 0
name_of_file = ''

transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

if DATASET == 0:
    ORIGINAL_DATA_DIR = '../data/fl27/original'
    CROPPED_DATA_DIR = '../data/fl27/images'
    ORIGINAL_ANNOTATION = '../data/fl27/annotation.txt'
    CROPPED_ANNOTATION = '../data/fl27/crop_annotation.txt'

    TRAIN_SET = '../annotations/trainset.txt'
    TEST_SET = '../annotations/testset.txt'

    LABELS = ['Adidas', 'Apple', 'BMW', 'Citroen', 'Cocacola', 'DHL', 'Fedex', 'Ferrari', 'Ford', 'Google', 'Heineken', 'HP', 'Intel',
              'McDonalds', 'Mini', 'Nbc', 'Nike', 'Pepsi', 'Porsche', 'Puma', 'RedBull', 'Sprite', 'Starbucks', 'Texaco', 'Unicef', 'Vodafone', 'Yahoo']


elif DATASET == 1:
    ORIGINAL_DATA_DIR = '../data/fl32/originals'
    CROPPED_DATA_DIR = '../data/fl32/images'
    ORIGINAL_ANNOTATION = '../data/fl32/annotation.txt'
    CROPPED_ANNOTATION = '../data/fl32/crop_annotation.txt'

    TRAIN_SET = '../annotations/trainset32.txt'
    TEST_SET = '../annotations/testset32.txt'
    LABELS = ['ferrari', 'ups', 'cocacola', 'guiness', 'adidas', 'aldi', 'texaco', 'nvidia', 'rittersport', 'paulaner', 'dhl', 'bmw', 'fosters', 'milka', 'starbucks', 'pepsi',
              'singha', 'apple', 'fedex', 'carlsberg', 'hp', 'chimay', 'google', 'tsingtao', 'corona', 'ford', 'esso', 'shell', 'stellaartois', 'becks', 'heineken', 'erdinger', 'nologo']
else:
    pass


def read_from_annotation(path):
    file = open(path, "r")
    content = file.readlines()
    new = [x.split(" ")[:-1] for x in content]
    return new


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


class MyDataset(Dataset):
    def __init__(self, txt_file, root, transform=None):
        self.txt_file = txt_file
        self.root = root
        self.transform = transform

    def __len__(self):
        return self.txt_file.shape[0]

    def __getitem__(self, id):
        img_name = os.path.join(self.root, self.txt_file[id][0])
        img = Image.open(img_name)
        if self.transform is not None:
            img = self.transform(img)
        logo = int(self.txt_file[id][1])
        return img, logo

# print(LABELS[32])
# print(len(LABELS))


def prepare_num_dataset(annotation_path, set_path):
    arr = read_from_annotation(set_path)
    out = []
    for item in arr:
        tmp = [item[0], LABELS.index(item[1].split('\n')[0])]
        out.append(tmp)
    out = np.array(out)
    return out


train_data = prepare_num_dataset(CROPPED_ANNOTATION, TRAIN_SET)
test_data = prepare_num_dataset(CROPPED_ANNOTATION, TEST_SET)
trainset = MyDataset(train_data, CROPPED_DATA_DIR, transform)
testset = MyDataset(test_data, CROPPED_DATA_DIR, transform)


batch_size = 100


train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=testset,
                                          batch_size=batch_size,
                                          shuffle=False)
print("Train size {} items and test size {} items".format(
    len(trainset), len(testset)))
print('Batch size: {}'.format(len(test_loader)))


num_epochs = 600
learning_rate = 0.001
momentum = 0.9

n_classes = len(LABELS)
if MODEL == 0:
    model = alexnet.AlexNet(n_classes)
    name_of_file = 'alexnet.txt'
model = model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), 0.001, momentum=momentum)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


f = open(name_of_file, 'w')
for epoch in range(num_epochs):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()

    end = time.time()
    for i, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images_var = to_var(images)
        labels_var = to_var(labels)

        output = model(images_var)
        loss = criterion(output, labels_var)

        prec1, prec5 = accuracy(output.data, labels.cuda(), topk=(1, 5))
        losses.update(loss.data[0], images.size(0))
        top1.update(prec1[0], images.size(0))
        top5.update(prec5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if (i+1) % 30 == 0:
            text = 'Epoch: [{0}][{1}/{2}],Time {batch_time.val:.3f} ({batch_time.avg:.3f}), Data {data_time.val:.3f} ({data_time.avg:.3f}),Loss {loss.val:.4f} ({loss.avg:.4f}), Prec@1 {top1.val:.3f} ({top1.avg:.3f}), Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch+1, i+1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5)
            f.write(str(epoch))
            print(text)
        if (epoch+1) % 50 == 0:
            torch.save(model.state_dict(), 'alexnet-ep-{}.pt'.format(epoch+1))

  #   batch_time_eval = AverageMeter()
  #   losses_eval = AverageMeter()
  #   top1_eval = AverageMeter()
  #   top5_eval = AverageMeter()
  #   model = torch.nn.DataParallel(model)
  #   model.eval()
  #   end_eval = time.time()

  #   for i, (images, labels) in enumerate(test_loader):
            # images_var = to_var(images)
            # labels_var = to_var(labels)

            # images_var = Variable(images, volatile=True)
            # labels_var = Variable(labels, volatile=True)

            # output = model(images_var)
            # loss = criterion(output, labels_var)

            # prec1, prec5 = accuracy(output.data, labels.cuda(), topk=(1, 5))
            # losses_eval.update(loss.data[0], images.size(0))
            # top1_eval.update(prec1[0], images.size(0))
            # top5_eval.update(prec5[0], images.size(0))

            # batch_time_eval.update(time.time() - end_eval)
            # end_eval = time.time()
            # if i % 5 == 0:
            # 	print('Test: [{0}/{1}]\t'
  #                 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
  #                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
  #                 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
  #                 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
  #                  i, len(test_loader), batch_time=batch_time_eval, loss=losses_eval,
  #                  top1=top1_eval, top5=top5_eval))
f.close()
