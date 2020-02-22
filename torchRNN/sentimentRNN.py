import argparse
import numpy as np
from string import punctuation
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def load_process_data(seq_length=200, split_frac=0.8, batch_size=50):

    with open('review_data/reviews.txt') as f1:
        reviews = f1.read()

    with open('review_data/labels.txt') as f2:
        labels = f2.read()

    # get rid of punctuation
    reviews = reviews.lower()
    all_text = ''.join([c for c in reviews if c not in punctuation])

    # split by new lines and spaces
    reviews_split = all_text.split('\n')
    all_text = ' '.join(reviews_split)

    # create a list of words
    words = all_text.split()

    # build a dictionary that maps words to integers
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: idx for idx, word in enumerate(vocab, 1)}

    reviews_ints = []
    for review in reviews_split:
        reviews_ints.append([vocab_to_int[word] for word in review.split()])

    # 1 = positive, 0 = negative label conversion
    labels_split = labels.split('\n')
    encoded_labels = np.array([1 if label == 'positive' else 0
                               for label in labels_split])

    non_zero_idx = [idx for idx, review in enumerate(reviews_ints)
                    if len(review) != 0]

    reviews_ints = [reviews_ints[idx] for idx in non_zero_idx]
    encoded_labels = np.array([encoded_labels[idx] for idx in non_zero_idx])

    features = pad_features(reviews_ints, seq_length=seq_length)

    split_idx = int(len(features) * split_frac)
    train_x, remaining_x = features[:split_idx], features[split_idx:]
    train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[
                                                       split_idx:]

    test_idx = int(len(remaining_x) * 0.5)
    val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
    val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

    # create tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_x),
                               torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(val_x),
                               torch.from_numpy(val_y))
    test_data = TensorDataset(torch.from_numpy(test_x),
                              torch.from_numpy(test_y))

    trainloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    validloader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    return trainloader, validloader, testloader, vocab_to_int


def pad_features(reviews_ints, seq_length):

    features = np.zeros((len(reviews_ints), seq_length), dtype=int)
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features


class SentimentRNN(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim,
                 hidden_dim, n_layers, drop_prob=0.5):

        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_size)

        self.dropout = nn.Dropout(0.3)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):

        batch_size = x.size(0)

        # embeddings and lstm_out
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)

        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size, device):

        weight = next(self.parameters()).data

        hidden = (
            weight.new(self.n_layers,
                       batch_size,
                       self.hidden_dim).zero_().to(device),
            weight.new(self.n_layers,
                       batch_size,
                       self.hidden_dim).zero_().to(device)
        )

        return hidden


def train(args, model, device, trainloader,
          validloader, optimizer, criterion):

    for epoch in range(args.epochs):

        train_loss = 0.0

        model.train()
        h = model.init_hidden(args.batch_size, device)
        for batch_idx, (data, labels) in enumerate(trainloader):

            h = tuple([each.data for each in h])
            data, labels = data.to(device), labels.to(device)

            model.zero_grad()

            output, h = model(data, h)
            loss = criterion(output.squeeze(), labels.float())
            train_loss += loss.item() * data.size(0)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{:05}/{} ({:.2f}%)]\tLoss: {:.6f}'
                      ''.format(epoch, batch_idx * len(data),
                                len(trainloader.dataset),
                                100 * batch_idx / len(trainloader),
                                loss.item()))

        train_loss /= len(trainloader.dataset)
        print('\nTrain set: Average loss: {:.6f}'
              ''.format(train_loss))

        model.eval()
        val_loss = 0.0
        val_h = model.init_hidden(args.batch_size, device)
        for batch_idx, (data, labels) in enumerate(validloader):

            val_h = tuple([each.data for each in val_h])
            data, labels = data.to(device), labels.to(device)

            output, val_h = model(data, val_h)
            loss = criterion(output.squeeze(), labels.float())

            val_loss += loss.item() * data.size(0)

        val_loss /= len(validloader.dataset)
        print('\nValidation set: Average loss: {:.6f}'
              ''.format(val_loss))


def test(args, model, testloader, device, criterion):

    test_loss = []
    num_correct = 0

    # initialise hidden state
    h = model.init_hidden(args.batch_size)

    model.eval()
    for batch_idx, (data, labels) in enumerate(testloader):

        h = tuple([each.data for each in h])
        data, labels = data.to(device), labels.to(device)

        output, h = model(data, h)

        loss = criterion(output.squeeze(), labels.float())
        test_loss += loss.item() * data.size(0)

        pred = torch.round(output.squeeze())

        correct_tensor = pred.eq(labels.float().view_as(pred))
        num_correct += torch.sum(correct_tensor).item()

    test_loss /= len(testloader.dataset)
    accuracy = num_correct / len(testloader.dataset)

    print('\nTest set: Average loss: {:.6f}\tAccuracy: {:.6f}'
          ''.format(test_loss, accuracy))


def predict(net, test_review, sequence_length, vocab_to_int, device):

    net.eval()
    test_ints = tokenize_review(test_review, vocab_to_int)
    seq_length = sequence_length
    features = pad_features(test_ints, seq_length)

    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(features)
    batch_size = feature_tensor.size(0)

    # initialize hidden state
    h = net.init_hidden(batch_size)

    feature_tensor = feature_tensor.to(device)

    output, h = net(feature_tensor, h)
    pred = torch.round(output.squeeze())
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))

    if pred.item() == 1:
        print("Positive review detected!")
    else:
        print("Negative review detected.")


def tokenize_review(test_review, vocab_to_int):
    
    test_review = test_review.lower()
    test_text = ''.join([c for c in test_review if c not in punctuation])
    test_words = test_text.split()

    test_ints = [vocab_to_int[word] for word in test_words]

    return test_ints


def main():

    # read command line arguments
    parser = argparse.ArgumentParser(
        description='Sentiment Analysis Example'
    )

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--testbatch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--seq-length', type=int, default=200, metavar='C',
                        help='sequence length (default: 200)')
    parser.add_argument('--hidden-dim', type=int, default=256, metavar='C',
                        help='hidden dim (default: 256)')
    parser.add_argument('--layers', type=int, default=2, metavar='C',
                        help='number of layers (default: 2)')
    parser.add_argument('--embed-length', type=int, default=400, metavar='C',
                        help='embedding length (default: 400)')
    parser.add_argument('--clip', type=float, default=5, metavar='C',
                        help='gradient clipping (default: 5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training (default: false)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='batch logging interval (defaul: 10)')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    trainloader, validloader, testloader, vocab_to_int = load_process_data(
        seq_length=args.seq_length, split_frac=0.8,  batch_size=args.batch_size
    )

    vocab_size = len(vocab_to_int) + 1
    output_size = 1
    hidden_dim = args.hidden_dim
    n_layers = args.layers
    embedding_dim = args.embed_length

    model = SentimentRNN(vocab_size, output_size,
                         embedding_dim, hidden_dim, n_layers).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train(args, model, device, trainloader, validloader, optimizer, criterion)
    test(args, model, testloader, device, criterion)

    test_review_neg = 'The worst movie I have seen; acting was terrible ' \
                      'and I want my money back. ' \
                      'This movie had bad acting and the dialogue was slow.'

    predict(model, test_review_neg, args.seq_length, vocab_to_int, device)


if __name__ == '__main__':
    main()
