import random
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, d = zip(*batch)
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(s_next, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


input_dim = 128
num_actions = 4
gamma = 0.99

q_net = QNetwork(input_dim, num_actions)
target_net = QNetwork(input_dim, num_actions)
target_net.load_state_dict(q_net.state_dict())

optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
replay = ReplayBuffer(10000)


def train_step(batch_size):
    if len(replay) < batch_size:
        return
    s, a, r, s_next, d = replay.sample(batch_size)
    q_values = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_net(s_next).max(1)[0]
        target = r + gamma * next_q * (1 - d)
    loss = F.mse_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


for _ in range(200):
    s = torch.randn(input_dim).tolist()
    a = random.randrange(num_actions)
    r = random.random()
    s_next = torch.randn(input_dim).tolist()
    done = random.choice([0, 1])
    replay.push(s, a, r, s_next, done)

for _ in range(50):
    train_step(32)

target_net.load_state_dict(q_net.state_dict())
