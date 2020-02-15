import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as datautils

from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR

# resolves issue with OpenMP on mac
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class ArchCifarCNN(nn.Module):

    def __init__(self):

        super().__init__()

        # load in resnet18 architecture
        self.resnet = models.resnet18(pretrained=False, num_classes=10)

        # alter conv1 so that input shapes match
        self.resnet.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet.maxpool = torch.nn.Identity()

    def forward(self, x):

        x = self.resnet(x)
        x = F.log_softmax(x, dim=1)

        return x


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
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.6f}, '
          'Accuracy: {}/{} ({:.0f}%)\n'
          ''.format(test_loss, correct, len(test_loader.dataset),
                    100 * correct / len(test_loader.dataset)))

    return test_loss


def main():

    # read command line arguments
    parser = argparse.ArgumentParser(
        description='Load Architecture CIFAR-10 CNN Example'
    )

    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 20)')
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
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # load / process data
    trainset = datasets.CIFAR10('./data',
                                train=True,
                                download=True,
                                transform=train_transform)

    testset = datasets.CIFAR10('./data',
                               train=False,
                               download=True,
                               transform=test_transform)

    trainloader = datautils.DataLoader(trainset,
                                       batch_size=args.batch_size,
                                       **kwargs)

    testloader = datautils.DataLoader(testset,
                                      batch_size=args.testbatch_size,
                                      **kwargs)

    # define model / optimizer / loss criterion etc.
    model = ArchCifarCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
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
            torch.save(model.state_dict(), 'arch_cifarCNN_checkpoint.pt')

        scheduler.step()

    # load best model and save
    if args.save_model:
        model.load_state_dict(torch.load('arch_cifarCNN_checkpoint.pt'))
        torch.save(model.state_dict(), 'arch_cifarCNN.pt')


if __name__ == '__main__':
    main()
