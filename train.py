import config
from RNN import RNN

import torch
from torch import nn

configuration = config.learning_config

def choose_device():

    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    return device

def train(dataset, X, y):

    number_of_outputs = len(dataset.get_target_names())
    input_seq = [torch.Tensor(i) for i in X]
    target_seq = [torch.Tensor([i]) for i in y]
    inout_seq = list(zip(input_seq, target_seq))

    device = choose_device()

    if configuration['classifier'] == 'RNN':
        model = RNN(configuration['RNN model settings'][0], number_of_outputs, configuration['RNN model settings'][2], configuration['RNN model settings'][3])

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
