import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(self, img_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, dim=-1)


class TextEncoder(nn.Module):
    def __init__(self, vocab_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vocab_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, dim=-1)


class CLIP(nn.Module):
    def __init__(self, img_dim, vocab_dim, embed_dim):
        super().__init__()
        self.image_encoder = ImageEncoder(img_dim, embed_dim)
        self.text_encoder = TextEncoder(vocab_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images, texts):
        img_feat = self.image_encoder(images)
        txt_feat = self.text_encoder(texts)
        scale = self.logit_scale.exp()
        logits = scale * img_feat @ txt_feat.t()
        return logits


def clip_loss(logits):
    n = logits.size(0)
    labels = torch.arange(n, device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) * 0.5


# dummy training step
device = "cpu"
model = CLIP(img_dim=1024, vocab_dim=300, embed_dim=256).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

images = torch.randn(32, 1024).to(device)
texts = torch.randn(32, 300).to(device)

logits = model(images, texts)
loss = clip_loss(logits)

optim.zero_grad()
loss.backward()
optim.step()
