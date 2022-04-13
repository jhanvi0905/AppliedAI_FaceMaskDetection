from sklearn.model_selection import KFold

from dataProcess import get_data_split, bar_graph
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")
import argparse


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),

            # nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.AvgPool2d(2, 2),
            # nn.BatchNorm2d(512),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        return self.network(x)

def intialize_optimizer(lr, model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer, criterion


def train_model_kfold(dataset, model, optimizer, criterion, epochs, batch_size):

    kfold = KFold(n_splits=10, shuffle=True)
    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        print('------------fold no---------{}----------------------'.format(fold))
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)

        train_model(trainloader, testloader, model, optimizer, criterion, epochs)

def train_model(dataset_train, dataset_val, model, optimizer, criterion, epochs):
    epochs = epochs
    running_loss_history = []
    running_corrects_history = []
    val_running_loss_history = []
    val_running_corrects_history = []

    for e in range(epochs):

        running_loss = []
        running_corrects = []
        val_running_loss = []
        val_running_corrects = []

        for inputs, labels in dataset_train:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss.append(loss.item())
            running_corrects.append(torch.sum(preds == labels.data))

        else:
            with torch.no_grad():
                for val_inputs, val_labels in dataset_val:
                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs, val_labels)

                    _, val_preds = torch.max(val_outputs, 1)
                    val_running_loss.append(val_loss.item())
                    val_running_corrects.append(torch.sum(val_preds == val_labels.data))

            epoch_loss = torch.mean(torch.FloatTensor(running_loss))
            epoch_acc = torch.mean(torch.FloatTensor(running_corrects))
            running_loss_history.append(epoch_loss)
            running_corrects_history.append(epoch_acc)

            val_epoch_loss = torch.mean(torch.FloatTensor(val_running_loss))
            val_epoch_acc = torch.mean(torch.FloatTensor(val_running_corrects))
            val_running_loss_history.append(val_epoch_loss)
            val_running_corrects_history.append(val_epoch_acc)
            print('epoch :', (e + 1))
            print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc.item()))
            print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc.item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", '--batch_size', type=int, default=100)
    parser.add_argument("-k", '--kfold', type=int, default=0)
    parser.add_argument("-ep", '--epochs', type=int, default=10)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-bias", "--bias", type=int, default=1)
    args = parser.parse_args()
    # bar_graph()
    model = CNN()
    print(model)
    optimizer, loss = intialize_optimizer(args.learning_rate, model)
    if args.kfold == 0:
        train, val = get_data_split("Final_Dataset_Project2/", args.batch_size, False, args.bias)
        train_model(train, val, model, optimizer, loss, args.epochs)
    else:
        dataset = get_data_split("Final_Dataset_Project2/", args.batch_size, True, args.bias)
        train_model_kfold(dataset, model, optimizer, loss, args.epochs, args.batch_size)
    torch.save(model, 'output_models/ep' + str(args.epochs) + 'bs' + str(args.batch_size) + '.h5')
