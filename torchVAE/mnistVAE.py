import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as datautils

from torchvision import datasets, transforms
from torchvision.utils import save_image


class VAE(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(args, model, device, trainloader, optimizer, epoch):

    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(trainloader):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()

        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100.0 * batch_idx / len(trainloader),
                loss.item() / len(data)))

    train_loss /= len(trainloader.dataset)

    print('\nTrain set - Epoch: {} Average loss: {:.4f}'
          ''.format(epoch, train_loss))


def test(args, model, device, testloader, epoch):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for i, (data, _) in enumerate(testloader):
            data = data.to(device)

            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            test_loss += loss.item()

            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(args.batch_size,
                                                         1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_'
                           + str(epoch) + '.png', nrow=n)

    test_loss /= len(testloader.dataset)
    print('Test set: Loss: {:.4f}'.format(test_loss))


def main():

    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--testbatch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training (default: false)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='logging interval (default: 10)')

    args = parser.parse_args()

    if not os.path.exists('results'):
        os.makedirs('results')

    # set training device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    trainset = datasets.MNIST('../data',
                              train=True,
                              download=True,
                              transform=transforms.ToTensor())

    testset = datasets.MNIST('../data',
                             train=False,
                             transform=transforms.ToTensor())

    trainloader = datautils.DataLoader(trainset,
                                       batch_size=args.batch_size,
                                       **kwargs)

    testloader = datautils.DataLoader(testset,
                                      batch_size=args.testbatch_size,
                                      **kwargs)

    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, trainloader, optimizer, epoch)
        test(args, model, device, testloader, epoch)

        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')


if __name__ == '__main__':
    main()
