import torch
from torch import nn


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

def predict(model, X):

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