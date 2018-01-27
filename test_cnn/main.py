import dataset as ds
import os
import model as cnn
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import optim
import matplotlib.pyplot as plt
import tool as tl2
from skimage import io
import warnings
import uuid
warnings.filterwarnings("ignore")

print(uuid.uuid4().hex)

def predict_func(model, loader, main_labels):
    print('Predicting process...')
    for i, sample in enumerate(loader):
        images = Variable(sample[0])
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        name_labels = []
        transofrmed = []
        for in_img, prediction in zip(sample[0], predicted):
            tmp = transforms.ToPILImage()
            conv = tmp(in_img)
            path_out = '/home/kamalkhan/Documents/DiplomaThesis/data/fl27/segmented_resized/predicted'
            f_name = main_labels[prediction] + '_' + str(uuid.uuid4().hex) + '.jpg'
            path = os.path.join(path_out, f_name)
            io.imsave(path, conv)
def main():
    #Path for model
    model_path = '/home/kamalkhan/Documents/DiplomaThesis/train_cnn/cnn-2.pt'
	# Get labels for showing results
    main_annotation = '/home/kamalkhan/Documents/DiplomaThesis/data/fl27/crop_annotation.txt'
    labels = tl2.prepare_labels(main_annotation)
    # Init of convolutional neural network
    CNN = cnn.CNN(n_classes = 27)
    # Loading end evaluating of pre-trained model
    CNN.load_state_dict(torch.load(model_path))
    CNN.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    # Path for segmented images
    segmented_img_path = '/home/kamalkhan/Documents/DiplomaThesis/data/fl27/segmented'
    # Path for segmented images dataset annotation
    segment_set_path = '/home/kamalkhan/Documents/DiplomaThesis/annotations/segmentset.txt'
    # Path for segmented, resized and combined images
    segmented_resized_img_path = '/home/kamalkhan/Documents/DiplomaThesis/data/fl27/segmented_resized/not_predicted'
    
    # Get segmented dataset and loading for dividing into batches
    dataset = ds.get_segmented_dataset(segmented_img_path, segment_set_path, segmented_resized_img_path)
    test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=10, shuffle=False)

    predict_func(CNN, test_loader, labels)

if __name__ == "__main__":
    main()