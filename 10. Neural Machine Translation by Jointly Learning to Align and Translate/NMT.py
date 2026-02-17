import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        h, _ = self.rnn(x)
        return h


class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.W = nn.Linear(enc_dim + dec_dim, dec_dim)
        self.v = nn.Linear(dec_dim, 1, bias=False)

    def forward(self, enc_out, dec_state):
        T = enc_out.size(1)
        dec_state = dec_state.unsqueeze(1).repeat(1, T, 1)
        energy = torch.tanh(self.W(torch.cat([enc_out, dec_state], dim=2)))
        scores = self.v(energy).squeeze(2)
        alpha = F.softmax(scores, dim=1)
        context = torch.sum(enc_out * alpha.unsqueeze(2), dim=1)
        return context


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim + enc_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, y, context, hidden):
        y = self.embedding(y)
        x = torch.cat([y, context.unsqueeze(1)], dim=2)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out.squeeze(1))
        return out, hidden
