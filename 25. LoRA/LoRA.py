import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=4, bias=True, merge_weights=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        self.merge_weights = merge_weights
        self.merged = False

        self.base = nn.Linear(in_features, out_features, bias=bias)
        for p in self.base.parameters():
            p.requires_grad = False

        if r > 0:
            self.A = nn.Parameter(torch.zeros(r, in_features))
            self.B = nn.Parameter(torch.zeros(out_features, r))
            nn.init.normal_(self.A, mean=0.0, std=0.02)
            nn.init.zeros_(self.B)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    def _delta_w(self):
        if self.r <= 0:
            return None
        return (self.B @ self.A) * self.scaling

    def merge(self):
        if self.merged or self.r <= 0:
            self.merged = True
            return
        with torch.no_grad():
            self.base.weight += self._delta_w()
        self.merged = True

    def unmerge(self):
        if not self.merged or self.r <= 0:
            self.merged = False
            return
        with torch.no_grad():
            self.base.weight -= self._delta_w()
        self.merged = False

    def forward(self, x):
        if self.r <= 0 or self.merge_weights or self.merged:
            return self.base(x)
        y0 = self.base(x)
        delta = F.linear(x, self._delta_w(), None)
        return y0 + delta


class TinySelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, lora_r=4, lora_alpha=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.wq = LoRALinear(d_model, d_model, r=lora_r, alpha=lora_alpha, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = LoRALinear(d_model, d_model, r=lora_r, alpha=lora_alpha, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        for p in self.wk.parameters():
            p.requires_grad = False
        for p in self.wo.parameters():
            p.requires_grad = False

    def forward(self, x, attn_mask=None):
        b, t, d = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(b, t, d)
        out = self.wo(out)
        return out


class TinyModel(nn.Module):
    def __init__(self, vocab_size=100, d_model=64, n_heads=4, lora_r=4, lora_alpha=4, n_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.attn = TinySelfAttention(d_model, n_heads, lora_r=lora_r, lora_alpha=lora_alpha)
        self.ln = nn.LayerNorm(d_model)
        self.cls = nn.Linear(d_model, n_classes)

        for p in self.emb.parameters():
            p.requires_grad = False
        for p in self.ln.parameters():
            p.requires_grad = False
        for p in self.cls.parameters():
            p.requires_grad = False

    def forward(self, token_ids):
        x = self.emb(token_ids)
        x = self.attn(x)
        x = self.ln(x)
        x = x.mean(dim=1)
        logits = self.cls(x)
        return logits


def make_synthetic_batch(batch_size, seq_len, vocab_size, threshold):
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    s = tokens.sum(dim=1)
    y = (s >= threshold).long()
    return tokens, y


def train_demo(device="cpu"):
    torch.manual_seed(7)

    vocab_size = 100
    d_model = 64
    n_heads = 4
    seq_len = 32
    batch_size = 64
    threshold = int(seq_len * (vocab_size - 1) * 0.50)

    model = TinyModel(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, lora_r=4, lora_alpha=4).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=2e-3)

    for step in range(200):
        tokens, y = make_synthetic_batch(batch_size, seq_len, vocab_size, threshold)
        tokens = tokens.to(device)
        y = y.to(device)

        logits = model(tokens)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 50 == 0:
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = (pred == y).float().mean().item()
            print(f"step {step + 1}  loss {loss.item():.4f}  acc {acc:.3f}")

    model.attn.wq.merge()
    model.attn.wv.merge()

    with torch.no_grad():
        tokens, y = make_synthetic_batch(batch_size, seq_len, vocab_size, threshold)
        tokens = tokens.to(device)
        y = y.to(device)
        logits = model(tokens)
        loss = F.cross_entropy(logits, y).item()
        acc = (logits.argmax(dim=1) == y).float().mean().item()
    print(f"merged inference  loss {loss:.4f}  acc {acc:.3f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_demo(device=device)
