import torch
from torch import nn
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler

import config
import random

configuration = config.learning_config

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_size = output_size
        self.input_size = input_size

        # Defining the layers
        # RNN Layer
        self._rnn = nn.RNN(input_size, hidden_dim, n_layers, nonlinearity=configuration["activation function"])
        # Fully connected layer
        self._fc = nn.Linear(hidden_dim, output_size)
        self._softmax = nn.Softmax(dim=2)
        self._estimator_type = 'classifier'
        self._device = self.choose_device()

    def get_params(self, deep=True):
        return {"hidden_dim": self.hidden_dim, "n_layers": self.n_layers, "output_size": self.output_size, "input_size" : self.input_size}

    def forward(self, x):
        seq_length = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(seq_length)
        #hidden = self.init_hidden(1)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self._rnn(x.view(1, seq_length, 1), hidden)
        #out, hidden = self._rnn(x.view(1, 1, -1), hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        #out = out.contiguous().view(-1, self._hidden_dim)
        out = self._fc(out)

        return out, hidden

    def init_hidden(self, seq_length):
        device = self._device
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, seq_length, self.hidden_dim).to(device)
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden

    def choose_device(self):
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device

    def fit(self, X, y):

        input_seq = [torch.Tensor(i) for i in X]
        target_seq = [torch.Tensor([i]) for i in y]
        inout_seq = list(zip(input_seq, target_seq))
        random.shuffle(inout_seq)                               #shuffle in case ordered data comes in

        criterion = nn.CrossEntropyLoss()

        if configuration["optimizer"] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=configuration["learning rate"])
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=configuration["learning rate"])

        for epoch in range(1, configuration["number of epochs"] + 1):
            optimizer.zero_grad()  # Clears existing gradients from previous epoch
            for seq, labels in inout_seq:
                #optimizer.zero_grad()  # Clears existing gradients from previous epoch
                # input_seq = input_seq.to(device)
                output, hidden = self(seq)

                output = output.to(self._device)
                last_output = output[-1][-1].view(1, -1)
                #last_output = output[-1].view(1, 6)

                label = torch.Tensor([labels.item()]).view(-1).long()
                loss = criterion(last_output, label)

                #label = torch.Tensor([labels.item()]).view(-1).long()
                #loss = criterion(output, label)
                loss.backward()  # Does backpropagation and calculates gradients
                optimizer.step()  # Updates the weights accordingly

            if epoch % 10 == 0:
                print('Epoch: {}/{}.............'.format(epoch, configuration["number of epochs"]), end=' ')
                print("Loss: {:.4f}".format(loss.item()))

        return self

    def predict(self, X):

        input_seq = [torch.Tensor(i) for i in X]
        pred = []

        for seq in input_seq:
            output, hidden = self(seq)
            output = output.to(self._device)

            prob = self._softmax(output)[-1][-1]

            # chose class that has highest probability
            try:
                if prob[0].item() > prob[1].item():
                    pred.append(0)
                else:
                    pred.append(1)
            except IndexError:
                pred.append(0)


        return pred

    def get_scaler(self):
        return self._zeromean