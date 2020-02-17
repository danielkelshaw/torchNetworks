import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as datautils

from torchvision import datasets, transforms

# resolves issue with OpenMP on mac
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Discriminator(nn.Module):

    def __init__(self, conv_dim=32):

        super().__init__()
        self.conv_dim = conv_dim

        self.conv1 = nn.Conv2d(3, conv_dim, 4, 2, 1)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(conv_dim * 2, conv_dim * 4, 4, 2, 1)

        self.fc = nn.Linear(conv_dim * 4 * 4 * 4, 1)

        self.batchnorm2 = nn.BatchNorm2d(conv_dim * 2)
        self.batchnorm3 = nn.BatchNorm2d(conv_dim * 4)

    def forward(self, x):

        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.batchnorm2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.batchnorm3(self.conv3(x)), 0.2)

        x = x.view(-1, self.conv_dim * 4 * 4 * 4)
        x = self.fc(x)

        return x


class Generator(nn.Module):

    def __init__(self, z_size, conv_dim=32):

        super().__init__()
        self.conv_dim = conv_dim

        self.fc = nn.Linear(z_size, conv_dim * 4 * 4 * 4)

        self.t_conv1 = nn.ConvTranspose2d(conv_dim * 4, conv_dim * 2, 4, 2, 1)
        self.t_conv2 = nn.ConvTranspose2d(conv_dim * 2, conv_dim, 4, 2, 1)
        self.t_conv3 = nn.ConvTranspose2d(conv_dim, 3, 4, 2, 1)

        self.batchnorm1 = nn.BatchNorm2d(conv_dim * 2)
        self.batchnorm2 = nn.BatchNorm2d(conv_dim)

    def forward(self, x):

        x = self.fc(x)
        x = x.view(-1, self.conv_dim * 4, 4, 4)

        x = F.relu(self.batchnorm1(self.t_conv1(x)))
        x = F.relu(self.batchnorm2(self.t_conv2(x)))
        x = F.relu(self.t_conv3(x))

        x = torch.tanh(x)

        return x


def scale(x, feature_range=(-1, 1)):
    f_min, f_max = feature_range
    x = x * (f_max - f_min) + f_min
    return x


def main():

    # read command line arguments
    parser = argparse.ArgumentParser(
        description='DCGAN Example'
    )

    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 20)')
    parser.add_argument('--testbatch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--conv-dim', type=float, default=32, metavar='LR',
                        help='conv_dim (default: 32)')
    parser.add_argument('--z-size', type=float, default=100, metavar='ZS',
                        help='size of latent space (default: 100)')
    parser.add_argument('--beta1', type=float, default=0.5, metavar='B1',
                        help='beta1 (default: 0.5)')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='B2',
                        help='beta2 (default: 0.999)')
    parser.add_argument('--smoothing', action='store_true', default=True,
                        help='applied label smoothing (default: true)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training (default: false)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='batch logging interval (defaul: 10)')
    parser.add_argument('--checkpoints', action='store_true', default=False,
                        help='save checkpoints in training (default: false)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='for saving the current model (default: false)')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform = transforms.ToTensor()

    trainset = datasets.SVHN('./data',
                             download=True,
                             transform=transform)

    dataloader = datautils.DataLoader(trainset,
                                      batch_size=args.batch_size,
                                      **kwargs)

    # define discriminator and generator
    D = Discriminator(args.conv_dim).to(device)
    G = Generator(z_size=args.z_size, conv_dim=args.conv_dim).to(device)

    # Create optimizers for the discriminator and generator
    d_optimizer = torch.optim.Adam(D.parameters(), args.lr,
                                   (args.beta1, args.beta2))

    g_optimizer = torch.optim.Adam(G.parameters(), args.lr,
                                   (args.beta1, args.beta2))

    criterion = nn.BCEWithLogitsLoss()

    real_labels = 0.9 if args.smoothing else 1.0
    fake_labels = 0.0

    losses = []
    for epoch in range(1, args.epochs + 1):

        for batch_idx, (real_images, _) in enumerate(dataloader):

            batch_size = real_images.size(0)
            real_images = scale(real_images)
            real_images = real_images.to(device)

            # TRAIN DISCRIMINATOR

            d_optimizer.zero_grad()

            # 1. Train with real images

            # Generate real images
            D_real = D(real_images)

            # Compute discriminator loss on real images
            labels = torch.full((batch_size,), real_labels, device=device)
            d_real_loss = criterion(D_real.squeeze(), labels)

            d_real_loss.backward()

            # 2. Train with fake images

            # Generate fake images
            z = torch.randn(batch_size, args.z_size, device=device)
            fake_images = G(z)

            # Compute  discriminator loss on fake images
            D_fake = D(fake_images)
            labels.fill_(fake_labels)
            d_fake_loss = criterion(D_fake.squeeze(), labels)
            d_fake_loss.backward()

            d_loss = d_real_loss + d_fake_loss

            d_optimizer.step()

            # TRAIN GENERATOR

            g_optimizer.zero_grad()

            # 1. Train with fake images and flipped labels

            # Generate fake images
            fake_images = G(z)
            D_fake = D(fake_images)

            # Compute generator loss on fake images
            labels.fill_(real_labels)
            g_loss = criterion(D_fake.squeeze(), labels)

            g_loss.backward()
            g_optimizer.step()

            if batch_idx % args.log_interval == 0:
                losses.append((d_loss.item(), g_loss.item()))

                print('Train Epoch: {:02} [{:05}/{} ({:.2f}%)]\t'
                      'Discriminator Loss: {:.6f}\t'
                      'Generator Loss: {:.6f}'
                      ''.format(epoch, batch_idx * len(real_images),
                                len(dataloader.dataset),
                                100 * batch_idx / len(dataloader),
                                d_loss.item(), g_loss.item()))


if __name__ == '__main__':
    main()
