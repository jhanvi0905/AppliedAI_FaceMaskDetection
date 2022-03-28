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
            nn.AvgPool2d(2, 2),
            nn.BatchNorm2d(256),

            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        return self.network(x)


def intialize_optimizer(lr, model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer, criterion


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
    parser.add_argument("-ep", '--epochs', type=int, default=10)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    args = parser.parse_args()
    bar_graph()
    train, test, val = get_data_split("classified/", args.batch_size)
    model = CNN()
    print(model)
    optimizer, loss = intialize_optimizer(args.learning_rate, model)
    train_model(train, val, model, optimizer, loss, args.epochs)
    torch.save(model, 'output_models/ep' + str(args.epochs) + 'bs' + str(args.batch_size) + '.h5')
