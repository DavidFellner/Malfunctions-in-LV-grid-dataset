import torch
from torch import nn
from sklearn.preprocessing import MaxAbsScaler
import random
import numpy as np
from math import cos, pi

import config


configuration = config.learning_config

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_size = output_size
        self.input_size = input_size

        self._rnn = nn.RNN(input_size, hidden_dim, n_layers, nonlinearity=configuration["activation function"])
        self._fc = nn.Linear(hidden_dim, output_size)
        self._softmax = nn.Softmax(dim=2)
        self._estimator_type = 'classifier'
        self._device = self.choose_device()

    def get_params(self, deep=True):
        return {"hidden_dim": self.hidden_dim, "n_layers": self.n_layers, "output_size": self.output_size, "input_size" : self.input_size}

    def forward(self, x):
        seq_length = len(x[0])

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(seq_length)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self._rnn(x, hidden)

        # feed output into the fully connected layer
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

    def fit(self, X, y, early_stopping=True, warm_up=True):

        self.early_stopping = early_stopping
        self.warm_up = warm_up

        mini_batch_size = configuration["mini batch size"]
        criterion = nn.CrossEntropyLoss()
        nominal_lr = configuration["learning rate"] * mini_batch_size  # linear scaling of LR
        lr = nominal_lr
        loss = 10000000000                                             #set initial dummy loss
        lrs = []
        losses = []

        for epoch in range(1, configuration["number of epochs"] + 1):

            zipped_X_y = list(zip(X, y))
            random.shuffle(zipped_X_y)              #randomly shuffle samples to have different mini batches between epochs
            X, y = zip(*zipped_X_y)
            X = np.array(X)
            y = list(y)

            mini_batches = X.reshape((int(len(X)/mini_batch_size), mini_batch_size, len(X[0])))
            mini_batch_targets = np.array(y).reshape(int(len(y) / mini_batch_size), mini_batch_size)

            input_seq = [torch.Tensor(i).view(len(i), -1, 1) for i in mini_batches]
            target_seq = [torch.Tensor([i]).view(-1).long() for i in mini_batch_targets]
            inout_seq = list(zip(input_seq, target_seq))

            try:
                optimizer, lr = self.control_learning_rate(lr=lr, loss=loss, losses=losses, nominal_lr=nominal_lr, epoch=epoch)
            except IndexError:
                optimizer = self.choose_optimizer(lr)
            lrs.append(lr)

            optimizer.zero_grad()  # Clears existing gradients from previous epoch

            for sequences, labels in inout_seq:               #do minibatch with more than 1 sample
                output, hidden = self(sequences)

                output = output.to(self._device)
                last_outputs = torch.stack([i[-1] for i in output])

                loss = criterion(last_outputs, labels)

                loss.backward()     # Does backpropagation and calculates gradients
                optimizer.step()    # Updates the weights accordingly

            losses.append(loss)

            if self.early_stopping:
                a = 1
                #self.predict()
                #use prediction versus y_test for fscore and stop training when no improvement


            if epoch % 10 == 0:
                print('Epoch: {}/{}.............'.format(epoch, configuration["number of epochs"]), end=' ')
                print("Loss: {:.4f}".format(loss.item()))

        return self, losses, lrs

    def predict(self, X):

        input_seq = [torch.Tensor(i).view(len(i), -1, 1) for i in X]
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

    def choose_optimizer(self, alpha=configuration["learning rate"]):
        if configuration["optimizer"] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=alpha)
        return optimizer

    def control_learning_rate(self, lr=None, loss=None, losses=None, epoch=None, nominal_lr=None):
        warm_up_share = configuration["percentage of epochs for warm up"] / 100
        if self.warm_up and epoch < int(warm_up_share * configuration["number of epochs"]):
            lr = nominal_lr * epoch / int((warm_up_share * configuration["number of epochs"]))
            optimizer = self.choose_optimizer(alpha=lr)
        elif self.warm_up and epoch >= int(warm_up_share * configuration["number of epochs"]):
            lr = nominal_lr * cos((epoch - int(warm_up_share * configuration["number of epochs"]))/(int((1-warm_up_share) * configuration["number of epochs"]))*(pi/2))     #choose inverse log or sth
            optimizer = self.choose_optimizer(alpha=lr)
        else:
            if losses[-1] > loss:
                lr = lr * 1.1
                optimizer = self.choose_optimizer(alpha=lr)
            elif losses[-1] <= loss:
                print('Loss goes up! Learning rate is decreased')
                lr = lr * 0.90
                optimizer = self.choose_optimizer(alpha=lr)
        return optimizer, lr



    def fit_scaler(self, X):
        X_zeromean = np.array([x - x.mean() for x in X])                        # deduct it's own mean from every sample
        maxabs_scaler = MaxAbsScaler().fit(X_zeromean)                          # fit scaler as to scale training data between -1 and 1

        return maxabs_scaler

    def preprocessing(self, X, scaler):
        X = scaler.transform(X)
