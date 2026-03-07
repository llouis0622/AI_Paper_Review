import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_ch, dim):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x


class ViT(nn.Module):
    def __init__(self, img_size, patch_size, in_ch, num_classes, dim, depth, heads, mlp_ratio):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch_size, in_ch, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, self.patch.num_patches + 1, dim))
        self.blocks = nn.Sequential(*[
            EncoderBlock(dim, heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])
