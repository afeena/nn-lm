import torch
import torch.nn as nn
import torch.nn.functional as F
import time
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
    def __init__(self, seq_length):
        self.corpus = IndexMap()
        self.model = None
        self.max_len = seq_length

    def build_vocab(self, filename, vocab = False):
        if vocab:
            with open(filename) as tf:
                for word in tf:
                        self.corpus.add_wrd(str.lower(word))
        else:
            with open(filename) as tf:
                for sent in tf:
                    for word in sent.split():
                        self.corpus.add_wrd(str.lower(word))

    def prepare_data(self, filename):
        data = []
        with open(filename) as tf:
            for sent in tf:
                vec = []
                for word in sent.split():
                    idx = self.corpus.get_idx_by_wrd(word)
                    vec.append(idx)
                tens = torch.LongTensor(vec)
                if len(vec) > self.max_len:
                    tens.narrow(0, 0, self.max_len)
                data.append(tens)
        data = pad_sequence(data).view(-1)
        return data

    def get_1h_vector(self, word):
        dim = len(self.corpus)
        vector = np.zeros(dim)
        vector[self.corpus.get_idx_by_wrd(word)] = 1


    def train(self,
              train_file,
              test_file,
              n_epoch = 1,
              batch_size = 32,
              hidden_size = 256,
              emb_dim = 100,
              learning_rate = 0.001
              ):

        lr = learning_rate
        self.model = LM(hidden_size, len(self.corpus), emb_dim, bs=batch_size)

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        train_data = self.prepare_data(train_file)
        test_data = self.prepare_data(test_file)

        num_batches = len(train_data) // (batch_size*self.max_len)

        total_loss = 0
        start_time = time.time()
        for epoch in range(0, n_epoch):
            for i in range(num_batches):
                batch = train_data[i*self.max_len:i*self.max_len+self.max_len*batch_size]
                target = train_data[i*self.max_len+1:i*self.max_len+1+self.max_len*batch_size]

                self.model.zero_grad()

                inpt = Variable(batch.view(batch_size, -1)).transpose(0,1)

                res_scores, hidden = self.model(inpt)
                loss = loss_function(res_scores.view(-1, len(self.corpus)), target)
                loss.backward(retain_graph=True)
                optimizer.step()

                total_loss+=loss.item()
                if i>0 and i%20 == 0:
                    end = time.time() - start_time
                    print("epoch {}/{} batch {}/{} PP: {} time {}".format(epoch, n_epoch, i, num_batches, math.exp(total_loss/i), end))
                    start_time = time.time()

            print("epoch {}/{} PP: {}".format(epoch, n_epoch, math.exp(total_loss / num_batches)))

        print("========== eval ============")
        self.model.eval()
        total_loss = 0
        self.model.init_hiddent()
        num_batches = len(test_data) // (batch_size * self.max_len)
        with torch.no_grad():
            for i in range(num_batches):
                batch = test_data[i * self.max_len:i * self.max_len + self.max_len * batch_size]
                target = test_data[i * self.max_len + 1:i * self.max_len + 1 + self.max_len * batch_size]

                inpt = Variable(batch.view(batch_size, -1)).transpose(0, 1)

                res_scores, hidden = self.model(inpt)
                loss = loss_function(res_scores, target)
                total_loss+=loss.item()
        print(math.exp(total_loss)/num_batches)




if __name__ == "__main__":
    trainer = LMTrainer(35)
    trainer.build_vocab("data/train.corpus.small")
    trainer.train("data/train.corpus.small", "data/test.corpus.small")