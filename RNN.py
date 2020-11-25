import torch
from torch import nn

import config

configuration = config.learning_config

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.Softmax(dim=1)
        self._estimator_type = 'classifier'

    def forward(self, x):
        seq_length = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(seq_length)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x.view(seq_length, 1, 1), hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, seq_length):
        device = self.choose_device()
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
        number_of_outputs = len(dataset.get_target_names())
        input_seq = [torch.Tensor(i) for i in X]
        target_seq = [torch.Tensor([i]) for i in y]
        inout_seq = list(zip(input_seq, target_seq))

        device = self.choose_device()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=configuration["learning rate"])

        for epoch in range(1, configuration["number of epochs"] + 1):
            for seq, labels in inout_seq:
                optimizer.zero_grad()  # Clears existing gradients from previous epoch
                # input_seq = input_seq.to(device)
                output, hidden = model(seq)

                output = output.to(device)
                extended_labels = torch.Tensor([labels.item() for i in output]).view(-1).long()
                loss = criterion(output, extended_labels)
                loss.backward()  # Does backpropagation and calculates gradients
                optimizer.step()  # Updates the weights accordingly

            if epoch % 10 == 0:
                print('Epoch: {}/{}.............'.format(epoch, configuration["number of epochs"]), end=' ')
                print("Loss: {:.4f}".format(loss.item()))

        return model

    def predict(self, X):

        device = choose_device()

        # One-hot encoding our input to fit into the model
        character = np.array([[char2int[c] for c in character]])
        character = one_hot_encode(character, dict_size, character.shape[1], 1)
        character = torch.from_numpy(character)
        character = character.to(device)

        out, hidden = model(character)

        prob = nn.functional.softmax(out[-1], dim=0).data
        # Taking the class with the highest probability score from the output
        char_ind = torch.max(prob, dim=0)[1].item()

        # assign probablities to output neurons
        '''out = self.softmax(out)

        #chose class that has highest probability most of the time as classification result
        count = torch.bincount(torch.argmax(out, 1))
        try:
            if count[0].item() > count[1].item():
                out = torch.Tensor([0])
            else:
                out = torch.Tensor([1])
        except IndexError:
            out = torch.Tensor([0])'''

        return int2char[char_ind], hidden