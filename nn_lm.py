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

    def init_hidden(self, bsz):
        return (Variable(torch.zeros([1, bsz, self.hidden_dim])),
                Variable(torch.zeros([1, bsz, self.hidden_dim])))

    def forward(self, sentence, hidden):
        embd = self.word_embd(sentence)
        out, hidden = self.lstm(embd, hidden)
        decoded = self.ff(out.view(out.size(0) * out.size(1), out.size(2)))
        return decoded.view(out.size(0), out.size(1), decoded.size(1)), hidden


class LMTrainer():
    def __init__(self, seq_length):
        self.corpus = IndexMap(start_idx=1)
        self.model = None
        self.max_len = seq_length
        self.loss_function = nn.CrossEntropyLoss()

    def build_vocab(self, filename, vocab = False):
        if vocab:
            with open(filename) as tf:
                for word in tf:
                        self.corpus.add_wrd(word)
        else:
            with open(filename) as tf:
                for sent in tf:
                    for word in sent.split():
                        self.corpus.add_wrd(str.lower(word))

    def prepare_data(self, filename):
        data = []
        start_tok = self.corpus.get_idx_by_wrd("<s>")
        end_tok = self.corpus.get_idx_by_wrd("</s>")
        with open(filename) as tf:
            for sent in tf:
                vec = [start_tok]
                for word in sent.split():
                    idx = self.corpus.get_idx_by_wrd(word)
                    vec.append(idx)
                vec.append(end_tok)

                if len(vec) > self.max_len:
                    vec = vec[:self.max_len-1]
                    vec.append(end_tok)

                data.append(vec)
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
              test_file,
              n_epoch = 1,
              batch_size = 32,
              hidden_size = 256,
              emb_dim = 100,
              learning_rate = 0.001
              ):

        lr = learning_rate
        self.model = LM(hidden_size, len(self.corpus), emb_dim, bs=batch_size)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        train_data = self.prepare_data(train_file)


        num_batches = len(train_data) // (batch_size)

        total_loss = 0
        start_time = time.time()
        hidden = self.model.init_hidden(batch_size)
        for epoch in range(0, n_epoch):
            for i in range(num_batches):
                batch = [torch.LongTensor(s[:-1]) for s in train_data[i*batch_size:i*batch_size+batch_size]]
                target = [torch.LongTensor(s[1:]) for s in train_data[i*batch_size:i*batch_size+batch_size]]

                hidden = self.repackage_hidden(hidden)
                self.model.zero_grad()

                inpt = Variable(pad_sequence(batch, batch_first=True, padding_value=0), requires_grad = False).transpose(0,1)
                trgt = Variable(pad_sequence(target, padding_value=0),requires_grad = False).view(-1)

                res_scores, hidden = self.model(inpt, hidden)
                loss = self.loss_function(res_scores.view(-1, len(self.corpus)), trgt)
                loss.backward(retain_graph=True)
                optimizer.step()

                total_loss+=loss.item()
                if i>0 and i%20 == 0:
                    end = time.time() - start_time
                    print("epoch {}/{} batch {}/{} PP: {} time {}".format(epoch, n_epoch, i, num_batches, math.exp(total_loss/i), end))
                    start_time = time.time()

            print("epoch {}/{} PP: {}".format(epoch, n_epoch, math.exp(total_loss / num_batches)))

    def eval(self, test_file, batch_size):
        test_data = self.prepare_data(test_file)
        self.model.eval()

        print("========== eval ============")

        total_loss = 0
        hidden = self.model.init_hidden(batch_size)
        num_batches = len(test_data) // batch_size
        with torch.no_grad():
            for i in range(num_batches):
                batch = [torch.LongTensor(s[:-1]) for s in test_data[i * batch_size:i * batch_size + batch_size]]
                target = [torch.LongTensor(s[1:]) for s in test_data[i * batch_size:i * batch_size + batch_size]]

                inpt = Variable(pad_sequence(batch, batch_first=True, padding_value=0), requires_grad=False).transpose(
                    0, 1)
                trgt = Variable(pad_sequence(target, padding_value=0), requires_grad=False).view(-1)

                res_scores, hidden = self.model(inpt, hidden)
                loss = self.loss_function(res_scores.view(-1, len(self.corpus)), trgt)
                total_loss += loss.item()
        print("test PP: ", math.exp(total_loss / num_batches))


    def generate_text(self, start_word = "<s>", num_words = 5):

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


    arg_parser.add_argument("--train", default="data/train.corpus.small")
    arg_parser.add_argument("--test", default="data/test.corpus.small")
    arg_parser.add_argument("--vocab", default = None)
    arg_parser.add_argument("--max-seq-length", default=50)
    arg_parser.add_argument("--bs", default=32)
    arg_parser.add_argument("--epochs", default=1)
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

    trainer.train(args.train, args.test, n_epoch=args.epochs, batch_size=args.bs,
                  hidden_size=args.hs, emb_dim=args.embd, learning_rate=args.lr)
    trainer.eval(args.test, args.bs)

    trainer.generate_text()