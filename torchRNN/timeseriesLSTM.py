import argparse
import numpy as np

import torch
import torch.nn as nn

# resolves issue with OpenMP on mac
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class SequenceLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)

        self.fc = nn.Linear(51, 1)

    def forward(self, x, future=0):

        outputs = []

        output = 0
        h_t1 = torch.zeros(x.size(0), 51, dtype=torch.float32)
        c_t1 = torch.zeros(x.size(0), 51, dtype=torch.float32)
        h_t2 = torch.zeros(x.size(0), 51, dtype=torch.float32)
        c_t2 = torch.zeros(x.size(0), 51, dtype=torch.float32)

        for idx, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            h_t1, c_t1 = self.lstm1(input_t, (h_t1, c_t1))
            h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
            output = self.fc(h_t2)
            outputs.append(output)
        for idx in range(future):  # if we should predict the future
            h_t1, c_t1 = self.lstm1(output, (h_t1, c_t1))
            h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
            output = self.fc(h_t2)
            outputs.append(output)

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


def main():

    # read command line arguments
    parser = argparse.ArgumentParser(
        description='Time Series Prediction Example'
    )

    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--lr', type=float, default=0.9, metavar='LR',
                        help='learning rate (default: 0.9)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='for saving the current model (default: false)')
    parser.add_argument('--plot-output', action='store_true', default=False,
                        help='for plotting output of model (default: false)')

    args = parser.parse_args()

    T = 20
    L = 1000
    N = 100

    data = torch.sin((torch.FloatTensor(range(L))
                      + torch.randint(-4 * T, 4 * T, (N, 1))) / T)

    input = data[3:, :-1]
    target = data[3:, 1:]
    test_input = data[:3, :-1]
    test_target = data[:3, 1:]

    # build the model
    model = SequenceLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):

        def closure():
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)

            print('Train Epoch: {:02}\tTrain Loss = {:.6f}'
                  ''.format(epoch, loss))

            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            future = 1000
            preds = model(test_input, future)
            test_loss = criterion(preds[:, :-future], test_target)

            y = preds.detach().numpy()

        print('Train Epoch: {:02}\tTest Loss = {:.6f}'
              '\n'.format(epoch, test_loss))

    if args.save_model:
        torch.save(model.state_dict(), 'timeseriesLSTM.pt')

    if args.plot_output:

        import matplotlib.pyplot as plt
        plt.figure(figsize=(30, 10))
        plt.title('Predict future values for time sequences'
                  '\n(Dashed lines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)],
                     color, linewidth=2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future),
                     yi[input.size(1):], color + ':', linewidth=2.0)

        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')

        plt.savefig('predicted.png')
        plt.close()


if __name__ == '__main__':
    main()
