from Game import Game
from Agent import Agent
from Network import Network

import copy
import numpy as np
import torch
import ale_py
import random
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


# --- SumTree for Prioritized Experience Replay ---
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write = 0
        self.size = 0

    def _propagate(self, idx, change):
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def _retrieve(self, idx, s):
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        self._propagate(idx, priority - self.tree[idx])
        self.tree[idx] = priority

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


# --- Prioritized Experience Replay Buffer ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6, beta_start=0.4, beta_frames=1_000_000):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.max_priority = 1.0

    def add_sample(self, state, action, reward, next_state, terminated):
        self.tree.add(self.max_priority, (state, action, reward, next_state, terminated))

    def get_samples(self, num):
        batch, indices, priorities = [], [], []
        segment = self.tree.total() / num

        for i in range(num):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            batch.append(data)

        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        probs = np.array(priorities) / self.tree.total()
        weights = (self.tree.size * probs) ** (-beta)
        weights /= weights.max()

        return batch, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(float(td_error)) + 1e-6) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    @property
    def size(self):
        return self.tree.size


# --- Create Game environment ---
Visable = False
render_mode = "human" if Visable else "rgb_array"

gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5", render_mode=render_mode, continuous=False)
if not Visable:
    env = RecordVideo(env, video_folder="videos", episode_trigger=lambda ep: ep % 100 == 0, name_prefix="breakoutv2")

game = Game(env)

# --- Create Network and Copy ---
network = Network(num_actions=4)
network2 = copy.deepcopy(network)
for p in network2.parameters():
    p.requires_grad = False

# --- Set up PyTorch ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
network.to(device=device)
network2.to(device=device)
print(f"Training on {device}")

# --- Create Agent, Buffer, Optimizer ---
agent = Agent(network, num_actions=4, epsilon=1.0, device=device)
buffer = PrioritizedReplayBuffer(capacity=1_000_000)
optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)


# --- Training Update Function ---
def train_step(batch, weights):
    states, actions, rewards, next_states, dones = zip(*batch)

    states      = torch.cat(states).to(device)
    next_states = torch.cat(next_states).to(device)
    actions = torch.tensor(actions, dtype=torch.long,    device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones   = torch.tensor(dones,   dtype=torch.float32, device=device)
    weights = weights.to(device)

    q_sa = network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Double DQN: online net selects action, target net evaluates it
        best_actions = network(next_states).argmax(dim=1)
        next_q = network2(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
        # gamma_n discounts over the full n-step horizon
        target = rewards + gamma_n * next_q * (1 - dones)

    td_errors = (q_sa - target).detach().cpu().numpy()

    # IS-weighted Huber loss — importance sampling corrects PER's sampling bias
    element_loss = torch.nn.functional.smooth_l1_loss(q_sa, target, reduction='none')
    loss = (weights * element_loss).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=10)
    optimizer.step()

    return loss.item(), td_errors


# --- Hyperparameters ---
epochs      = 70001
gamma       = 0.99
n_step      = 4
gamma_n     = gamma ** n_step   # discount applied at the bootstrap step
step_count  = 0
batch_size  = 32
warmup      = 10000
target_sync = 1000
train_every = 4


# --- Train Agent ---
for episode in range(epochs):

    state, info = game.reset()
    game_over = False
    episode_reward = 0.0
    n_step_buf = []

    while not game_over:

        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = game.make_move(action)

        game_over = terminated or truncated
        episode_reward += reward  # track unclipped reward for logging

        # clip reward to {-1, 0, +1} before storing
        clipped_reward = float(np.sign(reward))

        n_step_buf.append((state, action, clipped_reward, next_state, terminated))

        # once the window is full, compute n-step return for the oldest transition
        if len(n_step_buf) >= n_step:
            G = sum(gamma ** i * n_step_buf[i][2] for i in range(n_step))
            done_n = any(t[4] for t in n_step_buf)
            buffer.add_sample(n_step_buf[0][0], n_step_buf[0][1], G, n_step_buf[-1][3], done_n)
            n_step_buf.pop(0)

        state = next_state

        step_count += 1
        agent.epsilon = max(0.01, 1.0 - step_count / 2_000_000)

        if buffer.size >= warmup and step_count % train_every == 0:
            batch, indices, weights = buffer.get_samples(batch_size)
            _, td_errors = train_step(batch, weights)
            buffer.update_priorities(indices, td_errors)

        if step_count % target_sync == 0:
            network2.load_state_dict(network.state_dict())

    # flush remaining transitions at episode end (shorter n-step returns)
    while len(n_step_buf) > 0:
        G = sum(gamma ** i * n_step_buf[i][2] for i in range(len(n_step_buf)))
        done_n = any(t[4] for t in n_step_buf)
        buffer.add_sample(n_step_buf[0][0], n_step_buf[0][1], G, n_step_buf[-1][3], done_n)
        n_step_buf.pop(0)

    if not Visable and episode % 100 == 0:
        torch.save(network.state_dict(), f"weights/breakout-episode-{episode}.pt")
    print(f"episode {episode}  reward {episode_reward:.0f}  eps {agent.epsilon:.3f}")
