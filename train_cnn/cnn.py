import dataset as ds
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import optim

#Dataset Preparing
main_ann_path = '../data/fl27/crop_annotation.txt'
images_path = '../data/fl27/resized'
train_set_path = '../annotations/trainset.txt'
test_set_path = '../annotations/testset.txt'

trainset, testset = ds.get_dataset(main_ann_path,images_path,train_set_path,test_set_path)
print(len(trainset), len(testset))


TRAIN = False
TEST  = True


# Hyper Parameters
num_epochs = 1
batch_size = 40
learning_rate = 0.001
momentum = 0.9

n_classes = 27


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=trainset,
	batch_size=batch_size,
	shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=testset,
	batch_size=batch_size,
	shuffle=False)


class CNN(nn.Module):
	def __init__(self, n_classes):
		super(CNN, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(64, 192, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			)
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, n_classes),
		)
	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), 256 * 6 * 6)
		x = self.classifier(x)
		return x

cnn = CNN(n_classes)

if TRAIN:
	# Loss and Optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(cnn.parameters(), lr = learning_rate, momentum=momentum)

	# Train the Model
	for epoch in range(num_epochs):
		for i, (images, labels) in enumerate(train_loader):
			images = Variable(images)
			labels = Variable(labels)
			# Forward + Backward + Optimize
			optimizer.zero_grad()
			outputs = cnn(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			if (i+1) % 10 == 0:
				print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
					%(epoch+1, num_epochs, i+1, len(trainset)//batch_size, loss.data[0]))

	# Save the Trained Model
	torch.save(cnn.state_dict(), 'cnn-for-Shera.pt')

if TEST:
	cnn = CNN(n_classes)
	cnn.load_state_dict(torch.load('cnn-for-Shera.pt'))
	cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
	correct = 0
	total = 0
	for images, labels in test_loader:
		images = Variable(images)
		outputs = cnn(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum()

	print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))
