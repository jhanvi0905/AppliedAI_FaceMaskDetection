from torchvision.transforms.transforms import Grayscale
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import warnings
warnings.filterwarnings("ignore")
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
import splitfolders
import os

def load_data(path):
    reshape_size = torchvision.transforms.Resize((64, 64))
    data_type = torchvision.transforms.ToTensor()
    normalized_metrics = transforms.Normalize((0.5,), (0.5,))
    return ImageFolder(root = path,transform = torchvision.transforms.Compose([reshape_size, data_type, normalized_metrics]))


def training_loader(training_dataset,batch_size):
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    return training_loader


def validation_loader(validation_dataset, batch_size):
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    return validation_loader


def im_convert(tensor):
    image = tensor.clone().detach().numpy()
    print(image.shape)
    image = image.transpose(1, 2, 0)

    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)
    return image

def bar_graph():
    path = '/Users/manishsehgal/Desktop/COMP6721-AI/AppliedAI_FaceMaskDetection/classified/'
    list= []
    list.append(len(os.listdir(path+'cloth')))
    list.append(len(os.listdir(path + 'ffp2')))
    list.append(len(os.listdir(path + 'ffp2_valve')))
    list.append(len(os.listdir(path + 'surgical')))
    list.append(len(os.listdir(path + 'without_mask')))
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    var = ['cloth','ffp2','ffp2_valve','surgical','without_mask']
    ax.bar(var, list)
    plt.show()

def get_data_split(folder_path, batch_size):
    splitfolders.ratio(folder_path, output="output", seed=80, ratio=(.7, 0.2, 0.1))
    #train, test = train_test_split(dataset,test_size=0.25, random_state=30)
    dataset_train = load_data("output/train")
    dataset_test = load_data("output/test")
    dataset_val = load_data("output/val")
    train_set= training_loader(dataset_train, batch_size)
    validation_set = validation_loader(dataset_test, batch_size)
    test_set = validation_loader(dataset_val, batch_size)
    print("the length of train data: ", len(dataset_train))
    print("Length of test and validation data: ", len(dataset_val))
    data_iter = iter(train_set)
    images, labels = data_iter.next()
    fig = plt.figure(figsize=(25, 4))
    print("Instance of Loaded Samples")
    classes = ['Cloth Mask', 'FFP2 Mask','FFP2 Mask With Valve','Surgical Mask', 'Without Mask']
    
    for idx in np.arange(10):
        fig.add_subplot(2,10,idx+1)
        plt.imshow(im_convert(images[idx]))
    plt.title(classes[labels[idx].item()])

    return train_set, validation_set, test_set
