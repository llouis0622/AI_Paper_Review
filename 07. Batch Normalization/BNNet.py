import torch
import torch.nn as nn
import torch.optim as optim


class BNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


model = BNNet()
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

x = torch.randn(64, 1, 28, 28)
y = torch.randint(0, 10, (64,))

model.train()
optimizer.zero_grad()
out = model(x)
loss = criterion(out, y)
loss.backward()
optimizer.step()
