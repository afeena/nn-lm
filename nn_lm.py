import torch
import torch.nn as nn
import time
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import math
import argparse

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

        if torch.cuda.is_available():
            self.cuda()

    def init_hidden(self, bsz):
        return (Variable(torch.zeros([1, bsz, self.hidden_dim])),
                Variable(torch.zeros([1, bsz, self.hidden_dim])))

    def forward(self, sentence, hidden):
        embd = self.word_embd(sentence)
        out, hidden = self.lstm(embd, hidden)
        decoded = self.ff(out.view(out.size(0) * out.size(1), out.size(2)))
        return decoded.view(out.size(0), out.size(1), decoded.size(1)), hidden

    def save_model(self, path):
        torch.save(self, path)

    def load_model(self, path):
        model = torch.load(path)
        model.eval()
        return model



class LMTrainer():
    def __init__(self, seq_length):
        self.corpus = IndexMap(start_idx=0)
        self.model = None
        self.max_len = seq_length
        self.loss_function = nn.CrossEntropyLoss()

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
        data = torch.LongTensor()
        end_tok = self.corpus.get_idx_by_wrd("</s>")
        with open(filename) as tf:
            for sent in tf:
                vec = []
                for word in sent.split():
                    idx = self.corpus.get_idx_by_wrd(str.lower(word))
                    vec.append(idx)
                vec.append(end_tok)

                vec = torch.LongTensor(vec)
                data = torch.cat((data,vec))

        return data

    def get_1h_vector(self, word):
        dim = len(self.corpus)
        vector = np.zeros(dim)
        vector[self.corpus.get_idx_by_wrd(word)] = 1

    def repackage_hidden(self, h):
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)


    def train(self,
              train_file,
              n_epoch = 1,
              batch_size = 32,
              hidden_size = 256,
              emb_dim = 100,
              learning_rate = 0.01
              ):

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        lr = learning_rate
        self.model = LM(hidden_size, len(self.corpus), emb_dim, bs=batch_size)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        train_data = self.prepare_data(train_file)

        num_batches = train_data.size()[0] // (batch_size * self.max_len)

        total_loss = 0
        start_time = time.time()
        hidden = self.model.init_hidden(batch_size)
        for epoch in range(0, n_epoch):
            for i in range(num_batches):
                batch = train_data[i*batch_size:i*batch_size+self.max_len*batch_size]
                target = train_data[i*batch_size+1:i*batch_size+1+self.max_len*batch_size]

                hidden = self.repackage_hidden(hidden)
                self.model.zero_grad()

                inpt = batch.view(batch_size, -1).transpose(0,1).to(device=device)
                trgt = target.view(-1).to(device=device)

                res_scores, hidden = self.model(inpt, hidden)
                loss = self.loss_function(res_scores.view(-1, len(self.corpus)), trgt)
                loss.backward()
                optimizer.step()

                total_loss+=loss.data
                if i>0 and i%20 == 0:
                    end = time.time() - start_time
                    print("epoch {}/{} batch {}/{} PP: {} time {}".format(epoch, n_epoch, i, num_batches, math.exp(total_loss.item()/20), end))
                    start_time = time.time()
                    total_loss = 0
        self.model.save_model("models/rnn")

    def eval(self, test_file, batch_size):

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        test_data = self.prepare_data(test_file)
        self.model.eval()

        print("========== eval ============")

        total_loss = 0
        hidden = self.model.init_hidden(batch_size)
        num_batches = test_data.size()[0] // (batch_size * self.max_len)
        with torch.no_grad():
            for i in range(num_batches):
                batch = test_data[i * batch_size:i * batch_size + self.max_len * batch_size]
                target = test_data[i * batch_size + 1:i * batch_size + 1 + self.max_len * batch_size]

                inpt = batch.view(batch_size, -1).transpose(0, 1).to(device=device)
                trgt = target.view(-1).to(device=device)

                res_scores, hidden = self.model(inpt, hidden)
                loss = self.loss_function(res_scores.view(-1, len(self.corpus)), trgt)
                total_loss += loss.item()

        print(math.exp(total_loss /  num_batches))

        return total_loss / test_data.size()[0]


    def generate_text(self, start_word = "man", num_words = 5):

            res = [start_word]

            with torch.no_grad():
                hidden = self.model.init_hidden(1)

                input = torch.LongTensor([self.corpus.get_idx_by_wrd(start_word)]).unsqueeze(0)

                for i in range(num_words):
                    prediction, hidden = self.model(input, hidden)
                    prediction = prediction.squeeze().detach()
                    decoder_output = prediction
                    max_prob = 0
                    best_next = -1
                    for i,w in enumerate(decoder_output):
                        if w>max_prob:
                            best_next = i
                    res.append(best_next)
                    input = torch.LongTensor([best_next]).unsqueeze(0)

            for idx in res:
                print(self.corpus.get_wrd_by_idx(idx))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()


    arg_parser.add_argument("--train", default="data/ptb.train.txt")
    arg_parser.add_argument("--test", default="data/ptb.test.txt")
    arg_parser.add_argument("--vocab", default = None)
    arg_parser.add_argument("--max-seq-length", default=50)
    arg_parser.add_argument("--bs", default=32)
    arg_parser.add_argument("--epochs", default=10)
    arg_parser.add_argument("--nlayers", default=1)
    arg_parser.add_argument("--hs", default=256)
    arg_parser.add_argument("--embd", default=100)
    arg_parser.add_argument("--lr", default=0.001)

    args = arg_parser.parse_args()

    trainer = LMTrainer(args.max_seq_length)

    if args.vocab:
        trainer.build_vocab(args.vocab, vocab=True)
    else:
        trainer.build_vocab(args.train)

    trainer.train(args.train, n_epoch=args.epochs, batch_size=args.bs,
                  hidden_size=args.hs, emb_dim=args.embd, learning_rate=args.lr)
    trainer.eval(args.test, 10)


    trainer.generate_text()