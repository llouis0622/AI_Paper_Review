import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Linear(state_dim, 64)
        self.policy = nn.Linear(64, action_dim)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        return self.policy(x), self.value(x)


def compute_loss(model, states, actions, old_log_probs, returns, advantages, epsilon):
    logits, values = model(states)
    dist = torch.distributions.Categorical(logits=logits)
    log_probs = dist.log_prob(actions)
    ratio = torch.exp(log_probs - old_log_probs)

    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    policy_loss = -torch.mean(torch.min(ratio * advantages, clipped * advantages))
    value_loss = F.mse_loss(values.squeeze(), returns)
    entropy = dist.entropy().mean()

    return policy_loss + 0.5 * value_loss - 0.01 * entropy


state_dim = 8
action_dim = 4
model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

states = torch.randn(32, state_dim)
actions = torch.randint(0, action_dim, (32,))
old_log_probs = torch.randn(32)
returns = torch.randn(32)
advantages = torch.randn(32)

loss = compute_loss(model, states, actions, old_log_probs, returns, advantages, epsilon=0.2)

optimizer.zero_grad()
loss.backward()
optimizer.step()
