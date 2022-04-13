import os

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score,classification_report
import seaborn as s
from dataProcess import load_data, im_convert, validation_loader
import torch
import numpy as np
import matplotlib.pyplot as plt
from train_model import CNN

def testing_loader(testing_dataset):
    test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=256, shuffle=True)
    return test_loader

def generate_sample_preds(test_load, model, path):
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    print(labels)
    output = model(images)
    _, preds = torch.max(output, 1)
    print(preds)
    fig = plt.figure(figsize=(25, 4))
    classes = ['ClothMask', 'FFP2 Mask', 'FFP2 Mask With Valve', 'Surgical Mask', 'Without Mask']
    for idx in np.arange(5):
        ax = fig.add_subplot(10, 10, idx + 1, xticks=[], yticks=[])
        print(classes[preds[idx]])
        ax.set_title(classes[preds[idx]])
        plt.imsave(path + classes[preds[idx]] + str(idx) + '.png',
                   im_convert(images[idx]))
    Accuracy = accuracy_score(preds, labels)
    print("Accuracy of Current Batch: ", Accuracy)
    confusion = confusion_matrix(preds, labels)
    s.set(font_scale=1.5)
    s.heatmap(
            confusion_matrix(labels, preds),
            annot=True,
            annot_kws={"size": 16},
            cmap="Blues"
    )
    plt.imsave(path+'confusion_matrix.png', confusion)
    report = classification_report(labels, preds, labels=[0,1,2,3,4], target_names=classes)
    print(report)

if __name__== "__main__":

    test_loader = load_data("/Users/jhanviarora/Desktop/Project/output_fol2/test/")
    test_dataloader = testing_loader(test_loader)
    model = torch.load('/Users/jhanviarora/Desktop/Project/output_models/ep10bs100.h5', map_location=torch.device('cpu'))
    generate_sample_preds(test_dataloader, model, '/Users/jhanviarora/Desktop/Project/Output_Predictions/')