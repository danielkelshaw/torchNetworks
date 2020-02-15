import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as datautils

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class mnistNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)

        return output


def train(args, model, device, train_loader, optimizer, criterion, epoch):

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{:05}/{} ({:.2f}%)]\tLoss: {:.6f}'
                  ''.format(epoch, batch_idx * len(data),
                            len(train_loader.dataset),
                            100 * batch_idx / len(train_loader), loss.item()))


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

    print('\nTest set: Average loss: {:.6f}, '
          'Accuracy: {}/{} ({:.0f}%)\n'
          ''.format(test_loss, correct, len(test_loader.dataset),
                    100. * correct / len(test_loader.dataset)))

    return test_loss


def main():

    # read command line arguments
    parser = argparse.ArgumentParser(description='Neural Network Example')

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

    trainloader = datautils.DataLoader(trainset, batch_size=args.batch_size)
    testloader = datautils.DataLoader(testset, batch_size=args.testbatch_size)

    # define model / optimizer / loss criterion etc.
    model = mnistNN().to(device)
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
            torch.save(model.state_dict(), 'mnist_nn_checkpoint.pt')

        scheduler.step()

    # load best model and save
    if args.save_model:
        model.load_state_dict(torch.load('mnist_nn_checkpoint.pt'))
        torch.save(model.state_dict(), 'mnist_nn.pt')


if __name__ == '__main__':
    main()
