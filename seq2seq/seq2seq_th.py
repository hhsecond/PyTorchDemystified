import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from util import get_data, get_accuracy

en, fr = get_data()


def get_batches(sources, targets, batch_size):
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        yield sources_batch, targets_batch


epochs = 8
batch_size = 128
rnn_size = 256
num_layers = 2
encoding_embedding_size = 128
decoding_embedding_size = 128
learning_rate = 0.001
keep_probability = 0.5
display_step = 100


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size)

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
