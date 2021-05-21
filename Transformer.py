import torch
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import random
import numpy as np
import copy
import importlib
import math
import gc
import os

from experiment_config import experiment_path, chosen_experiment
spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)


configuration = config.learning_config

def choose_best(models_and_losses):
    index_best = [i[1] for i in models_and_losses].index(min([i[1] for i in models_and_losses]))
    epoch = index_best+1
    return models_and_losses[index_best], epoch

def save_model(model, epoch, loss):
    path = os.path.join(config.models_folder, configuration['classifier'])

    if not os.path.exists(path):
        os.makedirs(path)

    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(path, 'model.pth'))
    except TypeError:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict,
            'optimizer_state_dict': model.optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(path, 'model.pth'))

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
         pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
             output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Transformer(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(Transformer, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.optimizer = self.choose_optimizer(alpha=configuration["learning rate"] * configuration["mini batch size"])     # linear scaling of LR

        self.init_weights()
        self._device = self.choose_device()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        #nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        #src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)

        #gc.collect()
        return F.log_softmax(output, dim=-1)

    def fit(self, train_loader=None, test_loader=None, X_train=None, y_train=None, X_test=None, y_test=None, early_stopping=True, control_lr=None, prev_epoch=1, prev_loss=1):

        torch.cuda.empty_cache()
        self.early_stopping = early_stopping
        self.control_lr = control_lr

        if X_train and y_train:
            X = X_train
            y = y_train

        mini_batch_size = configuration["mini batch size"]
        criterion = nn.CrossEntropyLoss()
        nominal_lr = configuration["learning rate"] * mini_batch_size  # linear scaling of LR
        lr = nominal_lr
        loss = 10000000000                                             #set initial dummy loss
        lrs = []
        training_losses = []
        models_and_val_losses = []
        pause = 0                                                      # for early stopping

        if prev_epoch is None:
            prev_epoch = 1

        for epoch in range(prev_epoch, configuration["number of epochs"] + 1):

            if configuration["optimizer"] == 'SGD' and not epoch == prev_epoch:             #ADAM optimizer has internal states and should therefore not be reinitialized every epoch; only for SGD bc here changing the learning rate makes sense
                self.optimizer, lr = self.control_learning_rate(lr=lr, loss=loss, losses=training_losses, nominal_lr=nominal_lr, epoch=epoch)
            lrs.append(lr)

            if X_train and y_train:
                zipped_X_y = list(zip(X, y))
                random.shuffle(zipped_X_y)              #randomly shuffle samples to have different mini batches between epochs
                X, y = zip(*zipped_X_y)
                X = np.array(X)
                y = list(y)

                if len(X) % mini_batch_size > 0:               #drop some samples if necessary to fit with batch size
                    samples_to_drop = len(X) % mini_batch_size
                    X = X[:-samples_to_drop]
                    y = y[:-samples_to_drop]

                mini_batches = X.reshape((int(len(X) / mini_batch_size), mini_batch_size, len(X[0])))
                mini_batch_targets = np.array(y).reshape(int(len(y) / mini_batch_size), mini_batch_size)

                input_seq = [torch.Tensor(i).view(len(i), -1, 1) for i in mini_batches]
                target_seq = [torch.Tensor([i]).view(-1).long() for i in mini_batch_targets]
                inout_seq = list(zip(input_seq, target_seq))



                #optimizer.zero_grad()  # Clears existing gradients from previous epoch

                for sequences, labels in inout_seq:
                    labels = labels.to(self._device)
                    sequences = sequences.to(self._device)
                    self.optimizer.zero_grad()  # Clears existing gradients from previous batch so as not to backprop through entire dataset

                    #output = self(sequences, labels.float()).int()
                    output = self(sequences)

                    last_outputs = torch.stack([i[-1] for i in output])         #choose last output of timeseries (most informed output)
                    last_outputs = last_outputs.to(self._device)

                    loss = criterion(last_outputs, labels)

                    loss.backward()     # Does backpropagation and calculates gradients
                    self.optimizer.step()    # Updates the weights accordingly

                    gc.collect()
                    self.detach([sequences, labels])      #detach tensors from GPU to free memory

            elif train_loader and test_loader:
                import sys
                toolbar_width = len(train_loader)
                # setup toolbar
                print('Epoch completed:')
                sys.stdout.write("[%s]" % (" " * toolbar_width))
                sys.stdout.flush()
                sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

                for i, (sequences, labels, raw_seq) in enumerate(train_loader):
                    labels = labels.to(self._device)
                    sequences = sequences.to(self._device)
                    self.optimizer.zero_grad()  # Clears existing gradients from previous batch so as not to backprop through entire dataset
                    output = self(sequences.view(len(sequences), -1, 1))

                    last_outputs = torch.stack([i[-1] for i in output])         #choose last output of timeseries (most informed output)
                    last_outputs = last_outputs.to(self._device)

                    labels = torch.stack([i[-1] for i in labels]).long()
                    loss = criterion(last_outputs, labels)

                    loss.backward()     # Does backpropagation and calculates gradients
                    #torch.nn.utils.clip_grad_norm_(self.parameters(), configuration["gradient clipping"])       # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                    self.optimizer.step()    # Updates the weights accordingly

                    self.detach([last_outputs, sequences, labels])      #detach tensors from GPU to free memory


                    progress = (i+1) / len(train_loader)
                    sys.stdout.write("- %.1f%% " %(progress*100))
                    sys.stdout.flush()


                sys.stdout.write("]\n") # this ends the progress bar

            else:
                print('Either provide X and y or dataloaders!')

            if X_train and y_train:
                training_losses.append(loss)
                val_outputs = torch.stack([i[-1].view(-1) for i in self.predict(X_test)[1]]).to(self._device)
                val_loss = criterion(val_outputs, torch.Tensor([np.array(y_test)]).view(-1).long().to(self._device))
            else:
                training_losses.append(loss)
                pred, val_outputs, y_test = self.predict(test_loader=test_loader)
                val_outputs = torch.stack([i[-1] for i in val_outputs]).to(self._device)
                y_test = y_test.view(-1).long().to(self._device)
                val_loss = criterion(val_outputs, y_test).to(self._device)

            models_and_val_losses.append((copy.deepcopy(self.state_dict), val_loss.item()))

            if configuration["save_model"]:
                clf, ep = choose_best(models_and_val_losses)
                if ep == epoch:
                    save_model(self, epoch, val_loss.item())

            if self.early_stopping:
                try:
                    if abs(models_and_val_losses[-1][1] - models_and_val_losses[-2][1]) < 1*10**-6:
                        pause += 1
                        if pause == 5:
                            print('Validation loss has not changed for {0} epochs! Early stopping of training after {1} epochs!'.format(pause, epoch))
                            return models_and_val_losses, training_losses, lrs
                except IndexError:
                    pass

            if not configuration["cross_validation"] and epoch % 10 == 0:
                print('Epoch: {}/{}.............'.format(epoch, configuration["number of epochs"]), end=' ')
                print("Loss: {:.4f}".format(loss.item()))

        return models_and_val_losses, training_losses, lrs

    def predict(self, test_loader=None, X=None):

        if X is not None:
            input_sequences = torch.stack([torch.Tensor(i).view(len(i), -1) for i in X])

            input_sequences = input_sequences.to(self._device)
            outputs = self(input_sequences)

            last_outputs = torch.stack([i[-1] for i in outputs]).to(self._device)
            probs = nn.Softmax(dim=-1)(last_outputs)

            pred = torch.argmax(probs, dim=-1)  # chose class that has highest probability

            self.detach([input_sequences, outputs])
            return [i.item() for i in pred], outputs
        elif test_loader:
            pred = torch.Tensor()
            y_test = torch.Tensor()
            outputs_cumm = torch.Tensor()
            for i, (input_sequences, labels, raw_seq) in enumerate(test_loader):
                input_sequences = input_sequences.to(self._device)
                outputs = self(input_sequences.view(len(input_sequences), -1, 1))

                last_outputs = torch.stack([i[-1] for i in outputs]).to(self._device)
                probs = nn.Softmax(dim=-1)(last_outputs)

                outputs_cumm = torch.cat((outputs_cumm.to(self._device), outputs.float()), 0)   #
                pred = torch.cat((pred.to(self._device), torch.argmax(probs, dim=-1).float()), 0)  # chose class that has highest probability
                y_test = torch.cat((y_test, labels.float()), 0)   # chose class that has highest probability

                self.detach([input_sequences, outputs])
                if configuration["train test split"] <= 1:
                    share_of_test_set = len(test_loader)*configuration["train test split"]*labels.size()[0]
                else:
                    share_of_test_set = configuration["train test split"]
                if y_test.size()[0] >= share_of_test_set:           #to choose the test set size (memory issues!!)
                    break
            return [i.item() for i in pred], outputs_cumm, y_test

        else:
            print('Either provide X or a dataloader!')

    def choose_optimizer(self, alpha=configuration["learning rate"]):
        if configuration["optimizer"] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=alpha)
        return optimizer

    def control_learning_rate(self, lr=None, loss=None, losses=None, epoch=None, nominal_lr=None):
        warm_up_share = configuration["percentage of epochs for warm up"] / 100
        if self.control_lr == 'warm up' and epoch < int(warm_up_share * configuration["number of epochs"]):
            lr = nominal_lr * epoch / int((warm_up_share * configuration["number of epochs"]))
            optimizer = self.choose_optimizer(alpha=lr)
        elif self.control_lr == 'warm up' and epoch >= int(warm_up_share * configuration["number of epochs"]):
            lr = nominal_lr * (configuration["number of epochs"] - epoch) / int((1-warm_up_share) * configuration["number of epochs"])
            optimizer = self.choose_optimizer(alpha=lr)
        elif self.control_lr == 'LR controlled':
            if losses[-1] > loss:
                lr = lr * 1.1
                optimizer = self.choose_optimizer(alpha=lr)
            elif losses[-1] <= loss:
                lr = lr * 0.90
                optimizer = self.choose_optimizer(alpha=lr)
        else:
            lr = lr
            optimizer = self.choose_optimizer(alpha=lr)
        return optimizer, lr

    def preprocess(self, X_train, X_test):
        scaler = self.fit_scaler(X_train)
        X_train = self.preprocessing(X_train, scaler)
        X_test = self.preprocessing(X_test, scaler)
        return X_train, X_test

    def fit_scaler(self, X):
        X_zeromean = np.array([x - x.mean() for x in X])                        # deduct it's own mean from every sample
        maxabs_scaler = MaxAbsScaler().fit(X_zeromean)                          # fit scaler as to scale training data between -1 and 1
        return maxabs_scaler

    def preprocessing(self, X, scaler):
        X_zeromean = np.array([x - x.mean() for x in X])
        X = scaler.transform(X_zeromean)
        return X

    def score(self, y_test, y_pred):
        metrics = precision_recall_fscore_support(y_test, y_pred, average='macro')
        accuracy = accuracy_score(y_test, y_pred)
        return [accuracy, metrics]

    def get_params(self, deep=True):
        return {"dim_feedforward": self._dim_feedforward, "input_size": self._input_size}

    def choose_device(self):
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device

    def detach(self, inputs=[]):
        for i in inputs:
            try:
                torch.detach(i)
            except TypeError:
                for k in i:
                    torch.detach(k)
        torch.cuda.empty_cache()
        return