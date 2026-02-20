import torch
import torch.nn as nn


class GPTBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

    def forward(self, x, mask):
        a, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.ln1(x + a)
        f = self.ff(x)
        return self.ln2(x + f)


class GPT(nn.Module):
    def __init__(self, vocab, d=256, n=4, h=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, 512, d))
        self.blocks = nn.ModuleList([GPTBlock(d, h) for _ in range(n)])
        self.lm = nn.Linear(d, vocab)

    def forward(self, x):
        t = x.size(1)
        mask = torch.triu(torch.ones(t, t), diagonal=1).bool()
        h = self.emb(x) + self.pos[:, :t]
        for b in self.blocks:
            h = b(h, mask)
        return self.lm(h)


model = GPT(10000)
x = torch.randint(0, 10000, (8, 64))
y = model(x)
