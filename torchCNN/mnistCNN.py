import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as datautils

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class MnistCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout1d = nn.Dropout(0.5)
        self.dropout2d = nn.Dropout2d(0.25)

    def forward(self, x):

        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))

        x = self.dropout2d(x)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout1d(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)

        return output


def train(args, model, device, train_loader, optimizer, criterion, epoch):

    train_loss = 0.0

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item() * data.size(0)
        loss.backward()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{:05}/{} ({:.2f}%)]\tLoss: {:.6f}'
                  ''.format(epoch, batch_idx * len(data),
                            len(train_loader.dataset),
                            100 * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader.dataset)

    print('\nTrain set: Average loss: {:.6f}'
          ''.format(train_loss))


def test(model, device, test_loader, criterion):

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.6f}, '
          'Accuracy: {}/{} ({:.0f}%)\n'
          ''.format(test_loss, correct, len(test_loader.dataset),
                    100. * correct / len(test_loader.dataset)))

    return test_loss


def main():

    # read command line arguments
    parser = argparse.ArgumentParser(
        description='MNIST CNN Example'
    )

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--testbatch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='learning rate step, gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training (default: false)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='batch logging interval (defaul: 10)')
    parser.add_argument('--checkpoints', action='store_true', default=False,
                        help='save checkpoints in training (default: false)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='for saving the current model (default: false)')

    args = parser.parse_args()

    # set training device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # load / process data
    trainset = datasets.MNIST('./data',
                              train=True,
                              download=True,
                              transform=transform)

    testset = datasets.MNIST('./data',
                             train=False,
                             download=True,
                             transform=transform)

    trainloader = datautils.DataLoader(trainset,
                                       batch_size=args.batch_size,
                                       **kwargs)

    testloader = datautils.DataLoader(testset,
                                      batch_size=args.testbatch_size,
                                      **kwargs)

    # define model / optimizer / loss criterion etc.
    model = MnistCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.NLLLoss()
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # run training
    min_test_loss = np.Inf
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, trainloader, optimizer, criterion, epoch)
        test_loss = test(model, device, testloader, criterion)

        if args.checkpoints:
            if test_loss < min_test_loss:
                print('\nValidation Loss Decreased: {:.6f} -> {:.6f}\n'
                      ''.format(min_test_loss, test_loss))

            min_test_loss = test_loss
            torch.save(model.state_dict(), 'mnistCNN_checkpoint.pt')

        scheduler.step()

    # load best model and save
    if args.save_model:
        model.load_state_dict(torch.load('mnistCNN_checkpoint.pt'))
        torch.save(model.state_dict(), 'mnistCNN.pt')


if __name__ == '__main__':
    main()
