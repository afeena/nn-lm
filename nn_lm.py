import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from index_map import IndexMap
torch.manual_seed(1)

class LM(nn.Module):
    def __init__(self, hidden_dim, vocab_size, embd_dim):
        super(LM, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embd = nn.Embedding(vocab_size, embd_dim)

        self.lstm = nn.LSTM(embd_dim, hidden_dim)

        self.ff = nn.Linear(hidden_dim, vocab_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))


    def forward(self, sentence):
        embd = self.word_embd(sentence)
        out, self.hidden = self.lstm(embd, self.hidden)

class LMTrainer():
    def __init__(self):
        self.corpus = IndexMap()

    def read_corpus(self, filename):


        with open(filename) as tf:
            for sent in tf:
                for word in sent.split():
                    self.corpus.add_wrd(word)

    def get_1h_vector(self, word):
        dim = len(self.corpus)
        vector = np.zeros(dim)
        vector[self.corpus.get_idx_by_wrd(word)] = 1