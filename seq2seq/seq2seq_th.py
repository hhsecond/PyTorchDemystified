import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from util import get_data, get_accuracy


"""
get_data returns objects with
        id2vocab: dict{int:char}
        vocab2id: dict{char:int}
        text: list[str]
        text_as_ids: list[list[int]]
"""
en, fr = get_data()

use_cuda = torch.cuda.is_available()


class EncoderNetowrk(nn.Module):
    """
    Encoder network with embedding and GRU layer
    param: vocab_size
    param: hidden_size
    param: embedding_size -> size of embedding vector (300 normally)
    param: num_layers
    """

    def __init__(self, vocab_size, hidden_size, embedding_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        # TODO - add dropout
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers)

    def forward(self, inputs, hidden):
        embedded = self.embed(inputs)
        # TODO - check why both inputs to gru should be 3 dimensional
        outputs, hidden = self.gru(embedded, hidden)
        return outputs, hidden


class DecoderNetwork(nn.Module):
    """
    Decoder network with embedding, GRU and fullyconnected Softmax layer
    param: vocab_size
    param: hidden_size
    param: embedding_size -> size of embedding vector (300 normally)
    param: num_layers
    """

    def __init__(self, vocab_size, hidden_size, embedding_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        # TODO - add dropout
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)
        # TODO - checkout all the functions that has been used
        self.softmax = nn.LogSoftmax()

    def forward(self, inputs, hidden):
        embedded = self.embed(inputs)
        # TODO - check why both inputs to gru should be 3 dimensional
        outputs, hidden = self.gru(embedded, hidden)
        # TODO - check why 2 dimension not 3
        outputs = self.softmax(self.fc(outputs[0]))
        return outputs, hidden


encoder = EncoderNetowrk(100, 256, 300)
decoder = DecoderNetwork(100, 256, 300)
output, hidden = encoder(Variable(torch.LongTensor([[1]])), Variable(torch.zeros(1, 1, 256)))
dec_output, dec_hidden = decoder(Variable(torch.LongTensor([[1]])), hidden)
