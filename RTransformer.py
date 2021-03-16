import torch
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import random
import numpy as np
import importlib
import math, copy
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
    path = config.models_folder + configuration['classifier']

    if not os.path.exists(config.models_folder + configuration['classifier']):
        os.makedirs(config.models_folder + configuration['classifier']
                    )

    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'loss': loss,
        }, path + '\\model.pth')
    except TypeError:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict,
            'optimizer_state_dict': model.optimizer.state_dict(),
            'loss': loss,
        }, path + '\\model.pth')

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        d_ff = d_model * 4
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def attention(query, key, value, mask=None, dropout=None):
    """
        Compute 'Scaled Dot Product Attention'
        query, key, value : batch_size, n_head, seq_len, dim of space
    """

    d_k = query.size(-1)
    # scores: batch_size, n_head, seq_len, seq_len
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)


    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MHPooling(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        "Take in model size and number of heads."
        super(MHPooling, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        #auto-regressive
        attn_shape = (1, 3000, 3000)
        subsequent_mask =  np.triu(np.ones(attn_shape), k=1).astype('uint8')
        try:
            self.mask = (torch.from_numpy(subsequent_mask) == 0).unsqueeze(1).cuda()
        except RuntimeError:
            self.mask = (torch.from_numpy(subsequent_mask) == 0).unsqueeze(1)

    def forward(self, x):
        "Implements Figure 2"

        nbatches, seq_len, d_model = x.shape

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (x, x, x))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=self.mask[:,:, :seq_len, :seq_len],
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class LocalRNN(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_type, ksize, dropout):
        super(LocalRNN, self).__init__()
        """
        LocalRNN structure
        """
        self.ksize = ksize
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(output_dim, output_dim, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(output_dim, output_dim, batch_first=True)
        else:
            self.rnn = nn.RNN(output_dim, output_dim, batch_first=True)

        self.output = nn.Sequential(nn.Linear(output_dim, output_dim), nn.ReLU())

        # To speed up
        idx = [i for j in range(self.ksize-1,10000,1) for i in range(j-(self.ksize-1),j+1,1)]
        try:
            self.select_index = torch.LongTensor(idx).cuda()
            self.zeros = torch.zeros((self.ksize-1, input_dim)).cuda()
        except RuntimeError:
            self.select_index = torch.LongTensor(idx)
            self.zeros = torch.zeros((self.ksize-1, input_dim))

    def forward(self, x):
        nbatches, l, input_dim = x.shape
        x = self.get_K(x) # b x seq_len x ksize x d_model
        batch, l, ksize, d_model = x.shape
        h = self.rnn(x.view(-1, self.ksize, d_model))[0][:,-1,:]
        return h.view(batch, l, d_model)

    def get_K(self, x):
        batch_size, l, d_model = x.shape
        zeros = self.zeros.unsqueeze(0).repeat(batch_size, 1, 1)
        x = torch.cat((zeros, x), dim=1)
        key = torch.index_select(x, 1, self.select_index[:self.ksize*l])
        key = key.reshape(batch_size, l, self.ksize, -1)
        return key


class LocalRNNLayer(nn.Module):
    "Encoder is made up of attconv and feed forward (defined below)"
    def __init__(self, input_dim, output_dim, rnn_type, ksize, dropout):
        super(LocalRNNLayer, self).__init__()
        self.local_rnn = LocalRNN(input_dim, output_dim, rnn_type, ksize, dropout)
        self.connection = SublayerConnection(output_dim, dropout)

    def forward(self, x):
        "Follow Figure 1 (left) for connections."
        x = self.connection(x, self.local_rnn)
        return x



class Block(nn.Module):
    """
    One Block
    """
    def __init__(self, input_dim, output_dim, rnn_type, ksize, N, h, dropout):
        super(Block, self).__init__()
        self.layers = clones(
            LocalRNNLayer(input_dim, output_dim, rnn_type, ksize, dropout), N)
        self.connections = clones(SublayerConnection(output_dim, dropout), 2)
        self.pooling = MHPooling(input_dim, h, dropout)
        self.feed_forward = PositionwiseFeedForward(input_dim, dropout)

    def forward(self, x):
        n, l, d = x.shape
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = self.connections[0](x, self.pooling)
        x = self.connections[1](x, self.feed_forward)
        return x



class RTransformer(nn.Module):
    """
    The overal model
    """
    def __init__(self, d_model, rnn_type, ksize, n_level, n, h, dropout):
        super(RTransformer, self).__init__()
        N = n
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, dropout)

        layers = []
        for i in range(n_level):
            layers.append(
                Block(d_model, d_model, rnn_type, ksize, N=N, h=h, dropout=dropout))
        self.forward_net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.forward_net(x)
        return x

class RT(nn.Module):
    def __init__(self, input_size, d_model, output_size, h, rnn_type, ksize, n, n_level, dropout, emb_dropout):
        super(RT, self).__init__()
        self.encoder = nn.Linear(input_size, d_model)
        self.rt = RTransformer(d_model, rnn_type, ksize, n_level, n, h, dropout)
        self.linear = nn.Linear(d_model, output_size)
        self.sig = nn.Sigmoid()
        self._device = self.choose_device()
        self.optimizer = self.choose_optimizer()

    def forward(self, x):
        x = self.encoder(x)
        output = self.rt(x)
        output = self.linear(output).double()
        return self.sig(output)

    def fit(self, train_loader=None, test_loader=None, X_train=None, y_train=None, X_test=None, y_test=None, early_stopping=True, control_lr=None):

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

        for epoch in range(1, configuration["number of epochs"] + 1):

            try:
                self.optimizer, lr = self.control_learning_rate(lr=lr, loss=loss, losses=training_losses, nominal_lr=nominal_lr, epoch=epoch)
            except IndexError:
                self.optimizer = self.choose_optimizer(lr)
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
                    #torch.nn.utils.clip_grad_norm_(self.parameters(), configuration["gradient clipping"])       # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
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
                    torch.nn.utils.clip_grad_norm_(self.parameters(), configuration["gradient clipping"])       # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
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

            models_and_val_losses.append((copy.deepcopy(self.state_dict()), val_loss.item()))

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
            outputs= self(input_sequences)

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

                outputs_cumm = torch.cat((outputs_cumm, outputs), 0)   #
                pred = torch.cat((pred, torch.argmax(probs, dim=-1)), 0)   # chose class that has highest probability
                y_test = torch.cat((y_test, labels), 0)   # chose class that has highest probability

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