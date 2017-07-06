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


class EncoderNetowrk:
    """
    Encoder network with embedding and GRU layer
    param: input_size -> vocab size
    param: hidden_size
    param: embedding_size -> size of embedding vector (300 normally)
    param: num_layers
    """

    def __init__(self, input_size, hidden_size, embedding_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(input_size, embedding_size)
        # TODO - add dropout
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers)

    def forward(self, inputs, hidden):
        embedded = self.embed(inputs)
        outputs, hidden = self.gru(embedded, hidden)
        return outputs, hidden
