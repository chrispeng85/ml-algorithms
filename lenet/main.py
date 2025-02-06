import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
from torchmetrics import Accuracy



class LeNet(nn.Module):

    def __init__(self):

        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, padding = 2)

        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5)

        self.fc1 = nn.Linear(in_features = 16*5*5, out_features = 120)

        self.fc2 = nn.Linear(in_features = 120, out_features= 84)

        self.fc3 = nn.Linear(in_features = 84, out_features = 10)

    def forward(self, x): #forward pass

        x = F.relu(self.conv1(x))

        x = F.avg_pool2d(x, kernel_size = 2, stride = 2)

        x = F.relu(self.conv2(x))

        x = F.avg_pool2d(x, kernel_size = 2, stride = 2)

        x = x.view(-1, 16*5*5)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x 



def load_transform():  

    transform = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])

    traindata = torchvision.datasets.MNIST(

        root = './data',
        train = True,
        download = True,
        transform = transform 

    )

    train_size = int(0.9 * len(traindata))

    val_size = len(traindata) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset = traindata, lengths = [train_size, val_size])


    test_dataset = torchvision.datasets.MNIST(

        root = './data',
        train = False,
        download = True,
        transform = transform
    )

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = 4,
        shuffle = True,
        num_workers = 2
    )

    valloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = 4,
        shuffle = True,
        num_workers = 2

    )

    testloader = torch.utils.data.DataLoader(

        test_dataset,
        batch_size = 4,
        shuffle = False,
        num_workers = 2

    )

    return trainloader, valloader, testloader


def train(model, trainloader,valloader,  testloader, criterion, optimizer, num_epochs, device):

    for epoch in range(num_epochs):

        train_loss, train_acc = 0.0, 0.0
        accuracy = Accuracy(task = 'multiclass', num_classes = 10)

        for X, y in trainloader:
            X,y = X.to(device), y.to(device)

        model.train()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        train_loss += loss.item()

        acc = accuracy(y_pred, y)
        train_acc += acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(trainloader)
    train_acc /= len(trainloader)


    #validation

    val_loss, val_acc = 0.0, 0.0
    model.eval()
    with torch.inference_mode():
        for X, y in valloader:
            X,y = X.to(device), y.to(device)

            y_pred = model(y)
            val_loss += loss.item()

            acc = accuracy(y_pred, y)
            val_acc += acc

        val_loss /= len(valloader)
        val_acc /= len(valloader)


    print(f'epoch: {epoch} | train loss: {train_loss:.5f} | train acc: {train_acc:.5f} | val loss: {val_loss:.5f} | val acc: {val_acc: .5f}')


def main():

    model = LeNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

    trainloader, valloader, testloader = load_transform()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    train(model, trainloader, valloader, testloader, criterion, optimizer, 20, device)

if __name__ == "__main__":

    main()



'''

three components: convolution, pooling and nonlinear activation functions

convolution for feature extraction
pooling layer for subsampling




notes on conv2d:

    out = bias + /sum weight * input


    where * is the 2d cross-correlation operator.

    (f*g)(tau) = \int complex conjugate of f(t)  g(t+tau)  dt

    is signal f present in g given lag tau? if cross correlation between
    f and g given tau is large, then we have confidence in so

    when symmetric signals are involved, convolution and cross-correlation become the same thing.

    cross correlation gives a "scan" of the unit cell against lattice structure


    



'''




