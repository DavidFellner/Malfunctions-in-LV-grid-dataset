import os
from RNN import RNN
from LSTM import LSTM
from GRU import GRU
from Transformer import Transformer
from RTransformer import RT
import torch


def model_exists(full_path):
    return os.path.exists(full_path + ".model") and os.path.exists(full_path + ".optimizer")

def load_model(learning_config):
    path = learning_config['models_folder'] + '\\' + learning_config['classifier'] + '\\' + learning_config['classifier']
    if learning_config['classifier'] == 'RNN':
        load_saved = model_exists(path)
        model = RNN(learning_config['RNN model settings'][0],  learning_config['RNN model settings'][1],
                    learning_config['RNN model settings'][2], learning_config['RNN model settings'][3])
    elif learning_config['classifier'] == 'LSTM':
        model = LSTM(learning_config['LSTM model settings'][0],  learning_config['LSTM model settings'][1],
                     learning_config['LSTM model settings'][2], learning_config['LSTM model settings'][3])
    elif learning_config['classifier'] == 'GRU':
        model = GRU(learning_config['GRU model settings'][0],  learning_config['GRU model settings'][1],
                    learning_config['GRU model settings'][2], learning_config['GRU model settings'][3])
    elif learning_config['classifier'] == 'Transformer':
        model = Transformer(learning_config['Transformer model settings'][0],  learning_config['Transformer model settings'][1], learning_config['Transformer model settings'][2], learning_config['Transformer model settings'][3], learning_config['Transformer model settings'][4], learning_config['Transformer model settings'][5])
    elif learning_config['classifier'] == 'RTransformer':
        model = RT(learning_config['R-Transformer model settings'][0],  learning_config['R-Transformer model settings'][1], learning_config['R-Transformer model settings'][2], learning_config['R-Transformer model settings'][3], learning_config['R-Transformer model settings'][4], learning_config['R-Transformer model settings'][5], learning_config['R-Transformer model settings'][6], learning_config['R-Transformer model settings'][7], learning_config['R-Transformer model settings'][8], learning_config['R-Transformer model settings'][9])
    else:
        print('Invalid model type entered!')
        return None

    optimizer = model.choose_optimizer()
    device = model.choose_device()

    if load_saved:
        print('Loading model ..')
        model.load_state_dict(torch.load(path + ".model"))
        optimizer.load_state_dict(torch.load(path + ".optimizer"))

    model.to(device)

    return model

def export_model(model, learning_config):
    dummy_input = torch.randn(1, 672, 1)
    out = model(dummy_input)
    input_names = ["input"]  # + ["learned_%d" % i for i in range(3)]
    output_names = ["output"]
    name = learning_config['dataset'] + '.onnx'

    model.eval()
    torch.onnx.export(torch.jit.trace_module(model, {'forward': dummy_input}), dummy_input, name, example_outputs=out, export_params=True, verbose=True,
                      input_names=input_names, output_names=output_names)


def save_model(model, learning_config):
    path = learning_config['models_folder'] + '\\' + learning_config['classifier'] + '\\' + learning_config['classifier']

    if not os.path.exists(learning_config['models_folder'] + '\\' + learning_config['classifier'] + '\\'):
        os.makedirs(learning_config['models_folder'] + '\\' + learning_config['classifier'] + '\\')

    torch.save(model.state_dict(), path + ".model")
    torch.save(model.optimizer.state_dict(), path + ".optimizer")
