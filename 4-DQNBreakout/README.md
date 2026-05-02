# Model 4 — Dueling Double DQN with Prioritized Experience Replay

Atari Breakout agent trained from raw pixels using a Dueling Double DQN with Prioritized Experience Replay and n-step returns.

## Evaluation Results (50 games, ε = 0.05)

| Metric | Score |
|--------|-------|
| Average | 27.9 |
| Best | 80 |
| Worst | 11 |
| Std dev | 12.1 |

## How It Works

The agent observes raw Atari frames and learns to play Breakout entirely from pixels. A stack of 4 preprocessed grayscale frames is fed into a convolutional neural network that outputs Q-values — estimates of expected future reward — for each of the 4 possible actions (NOOP, FIRE, LEFT, RIGHT).

### Preprocessing Pipeline (`Game.py`)

Each observation is processed before being fed to the network:

1. **Grayscale** — RGB channels are averaged to a single channel
2. **Resize** — frame is scaled to 84×84
3. **Frame stacking** — the last 4 frames are stacked as separate input channels, giving the network temporal context (ball velocity and direction)
4. **Frame skip** — the selected action is repeated for 4 environment steps, accumulating reward. This reduces computation and aligns with natural human reaction time
5. **Auto-FIRE** — when a life is lost, the FIRE action is issued automatically to relaunch the ball without wasting training steps on a stalled game

### Network Architecture — Dueling DQN (`Network.py`)

```
Input: (B, 4, 84, 84) — batch of stacked grayscale frames

Shared CNN Trunk:
  Conv2d(4 → 32, kernel=8, stride=4)   → (B, 32, 20, 20)
  ReLU
  Conv2d(32 → 64, kernel=4, stride=2)  → (B, 64, 9, 9)
  ReLU
  Conv2d(64 → 64, kernel=3, stride=1)  → (B, 64, 7, 7)
  ReLU
  Flatten                               → (B, 3136)

Value Head:     Linear(3136→512) → ReLU → Linear(512→1)   = V(s)
Advantage Head: Linear(3136→512) → ReLU → Linear(512→4)   = A(s, a)

Output: Q(s, a) = V(s) + ( A(s, a) − mean_a A(s, a) )
```

The dueling architecture separates "how good is this state?" (V) from "which action is best?" (A). In Breakout, most frames have similar action values — the ball is mid-flight and any action is roughly equivalent. Routing state value through a dedicated head lets the network learn V efficiently without contaminating it with action noise.

### Algorithm Improvements

This model stacks four improvements on top of basic DQN:

#### 1. Double DQN

Standard DQN uses the same network to select AND evaluate the next action, which systematically overestimates Q-values. Double DQN decouples the two:

```
best_action = argmax_a  online_net(s')
target      = r + γⁿ * target_net(s', best_action)
```

The online network picks the action; the frozen target network scores it. This removes the max-bias that inflates Q-value estimates in vanilla DQN.

#### 2. Prioritized Experience Replay (PER)

Uniform replay wastes capacity on transitions the agent already handles well. PER uses a SumTree to sample proportionally to TD error:

```
priority = ( |TD error| + ε )^α      (α = 0.6)
```

New transitions get maximum priority and are visited frequently until the error shrinks. Importance sampling (IS) weights correct the resulting distributional bias, with β annealing from 0.4 → 1.0 over 1M steps so corrections become exact as training matures.

#### 3. N-step Returns

Instead of bootstrapping on just the next state, 4-step returns are accumulated before storing into the replay buffer:

```
G_t = r_t + γ r_{t+1} + γ² r_{t+2} + γ³ r_{t+3}     (γ = 0.99, n = 4)
```

This propagates reward signal further back in time with each stored transition, speeding up learning when good outcomes are separated from the actions that caused them by many frames.

#### 4. Target Network

A frozen copy of the online network is maintained and synced every 1000 steps. Without it, the bootstrap target shifts on every update — the network chases a moving reference and training diverges or oscillates.

### Training Details

| Hyperparameter | Value |
|---|---|
| Episodes | 70,001 |
| Batch size | 32 |
| Replay capacity | 1,000,000 |
| Warmup steps | 10,000 |
| Train frequency | every 4 steps |
| Target sync | every 1,000 steps |
| Optimizer | Adam (lr = 1e-4) |
| Epsilon schedule | 1.0 → 0.01 over 2M steps |
| Gamma (γ) | 0.99 |
| N-step | 4 |
| Loss | Huber (smooth L1) × IS weights |
| Gradient clipping | max norm = 10 |
| Reward clipping | sign(r) ∈ {−1, 0, +1} |
| Device | MPS (Apple Silicon) / CPU fallback |

### Full Update Rule

```
loss = mean( w_i · Huber( Q(s, a; θ),  r + γ⁴ · Q(s', argmax_a Q(s'; θ); θ⁻) ) )

θ   = online network weights
θ⁻  = target network weights (frozen, synced every 1000 steps)
w_i = IS weight correcting PER's non-uniform sampling bias
```

## File Overview

| File | Purpose |
|------|---------|
| `Network.py` | Dueling DQN architecture (shared CNN trunk, value + advantage heads) |
| `Agent.py` | ε-greedy action selection |
| `Game.py` | Atari env wrapper — grayscale, resize, frame stacking, frame skip, auto-FIRE |
| `Train.py` | Training loop, SumTree PER buffer, Double DQN update, n-step returns |

## What's Next

- **Noisy nets** — replace ε-greedy with learned parameter noise in FC layers for more targeted exploration
- **Distributional RL (C51)** — model the full return distribution instead of its expectation; richer signal for the optimizer
- **Longer GPU training** — 70k episodes on Apple MPS is still proof-of-concept scale; stable tunnelling strategy needs roughly 10× more experience
