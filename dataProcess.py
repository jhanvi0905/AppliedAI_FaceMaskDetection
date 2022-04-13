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
    reshape_size = torchvision.transforms.Resize((128, 128))
    data_type = torchvision.transforms.ToTensor()
    normalized_metrics = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
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
    path = '/Users/jhanviarora/Desktop/Project/classified/'
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

def get_data_split(folder_path, batch_size, kfold = True, bias=1):
    if not kfold:
        if bias == 1:
            extra_fol1 = '/Female/'
            extra_fol2 = '/Male/'
        else:
            extra_fol1 = '/Old/'
            extra_fol2 = '/Young/'
        splitfolders.ratio(folder_path+ extra_fol1, output="output_fol1", seed=80, ratio=(.7, 0.2, 0.1))
        splitfolders.ratio(folder_path + extra_fol2, output="output_fol2", seed=80, ratio=(.7, 0.2, 0.1))
        dataset_train_fol1 = load_data("output_fol1/train")
        dataset_val_fol1 = load_data("output_fol1/val")
        dataset_train_fol2 = load_data("output_fol2/train")
        dataset_val_fol2 = load_data("output_fol2/val")
        total_train_dataset = torch.utils.data.ConcatDataset([dataset_train_fol2, dataset_train_fol1])
        total_val_dataset = torch.utils.data.ConcatDataset([dataset_val_fol1, dataset_val_fol2])
        train_set = training_loader(total_train_dataset, batch_size)
        validation_set = validation_loader(total_val_dataset, batch_size)
        print("the length of train data: ", len(total_train_dataset))
        print("Length of test: ", len(total_val_dataset))
        data_iter = iter(train_set)
        images, labels = data_iter.next()
        fig = plt.figure(figsize=(25, 4))
        print("Instance of Loaded Samples")
        classes = ['Cloth Mask', 'FFP2 Mask','FFP2 Mask With Valve','Surgical Mask', 'Without Mask']
        for idx in np.arange(10):
            fig.add_subplot(2,10,idx+1)
            plt.imshow(im_convert(images[idx]))
            plt.title(classes[labels[idx].item()])
        return train_set, validation_set
    else:
        return load_data(folder_path)



