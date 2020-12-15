import torch
from torch import nn
import importlib

from experiment_config import experiment_path, chosen_experiment
spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

configuration = config.learning_config

class Transformer(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Transformer, self).__init__()

        self._hidden_dim = hidden_dim
        self._n_layers = n_layers
        self._output_size = output_size
        self._input_size = input_size

        # Defining the layers
        # RNN Layer
        self._rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layer
        self._fc = nn.Linear(hidden_dim, output_size)
        self._softmax = nn.Softmax(dim=1)
        self._estimator_type = 'classifier'
        self._device = self.choose_device()

    def get_params(self, deep=True):
        return {"hidden_dim": self._hidden_dim, "n_layers": self._n_layers, "output_size": self._output_size, "input_size" : self._input_size}

    def forward(self, x):
        seq_length = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(seq_length)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self._rnn(x.view(seq_length, 1, 1), hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self._hidden_dim)
        out = self._fc(out)

        return out, hidden

    def init_hidden(self, seq_length):
        device = self._device
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self._n_layers, seq_length, self._hidden_dim).to(device)
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

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=configuration["learning rate"])

        for epoch in range(1, configuration["number of epochs"] + 1):
            for seq, labels in inout_seq:
                optimizer.zero_grad()  # Clears existing gradients from previous epoch
                # input_seq = input_seq.to(device)
                output, hidden = self(seq)

                output = output.to(self._device)
                last_output = output[-1].view(1, 2)

                label = torch.Tensor([labels.item() ]).view(-1).long()
                loss = criterion(last_output, label)
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
            prob = self._softmax(output)[-1]

            # chose class that has highest probability
            try:
                if prob[0].item() > prob[1].item():
                    pred.append(0)
                else:
                    pred.append(1)
            except IndexError:
                pred.append(0)


        return pred