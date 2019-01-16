import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import math


from index_map import IndexMap
torch.manual_seed(1)

class LM(nn.Module):
    def __init__(self, hidden_dim, vocab_size, embd_dim, bs=32, n_layers = 1):
        super(LM, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.word_embd = nn.Embedding(vocab_size, embd_dim)

        self.lstm = nn.LSTM(embd_dim, hidden_dim)

        self.ff = nn.Linear(hidden_dim, vocab_size)
        self.hidden = self.init_hidden(bs)

    def init_hidden(self, bsz):
        return (Variable(torch.zeros([1, bsz, self.hidden_dim])),
                Variable(torch.zeros([1, bsz, self.hidden_dim])))

    def forward(self, sentence):
        embd = self.word_embd(sentence)
        out, self.hidden = self.lstm(embd, self.hidden)
        decoded = self.ff(out.view(out.size(0) * out.size(1), out.size(2)))
        return decoded.view(out.size(0), out.size(1), decoded.size(1)), self.hidden


class LMTrainer():
    def __init__(self):
        self.corpus = IndexMap()
        self.model = None
        self.input = []
        self.max_len = 0

    def read_corpus(self, filename):
        with open(filename) as tf:
            for sent in tf:
                vec = []
                for word in sent.split():
                    idx = self.corpus.add_wrd(str.lower(word))
                    vec.append(idx)
                self.input.append(vec)
                if len(vec)>self.max_len:
                    self.max_len = len(vec)


    def get_1h_vector(self, word):
        dim = len(self.corpus)
        vector = np.zeros(dim)
        vector[self.corpus.get_idx_by_wrd(word)] = 1


    def train(self, n_epoch = 1):
        b_size = 32
        self.model = LM(256, len(self.corpus), 100, bs=32)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        num_batches = len(self.input) // b_size
        for epoch in range(0, n_epoch):
            for i in range(num_batches):
                batch = [torch.LongTensor(sent) for sent in self.input[i*b_size:i*b_size+b_size]]
                target = [torch.LongTensor(sent) for sent in self.input[i*b_size+1:i*b_size+b_size+1]]
                
                self.model.zero_grad()

                inpt = Variable(pad_sequence(batch, batch_first=True).transpose(0,1))
                trgt = Variable(pad_sequence(target, batch_first=True).view(-1))

                res_scores, hidden = self.model(inpt)
                loss = loss_function(res_scores.view(-1, len(self.corpus)), trgt)
                loss.backward(retain_graph=True)
                optimizer.step()
                print("epoch {}/{} batch {}/{} PP: {}".format(epoch, n_epoch, i, num_batches, math.exp(loss.item())))



if __name__ == "__main__":
    trainer = LMTrainer()
    trainer.read_corpus("data/train.corpus")
    trainer.train(1)