import torch
from collections import deque
import gymnasium as gym

class Game(gym.Wrapper):
    def __init__(self, env, frame_skip=4):
        super().__init__(env)
        self.frames = deque(maxlen=4)
        self.lives = 5
        self.frame_skip = frame_skip

    def stack_screen(self):
        return torch.stack(list(self.frames)).unsqueeze(0)

    def screen_tensor(self, obs):
        obs = torch.from_numpy(obs).float() / 255.0
        obs = obs.mean(dim=2)
        obs = obs.unsqueeze(0).unsqueeze(0)
        obs = torch.nn.functional.interpolate(obs, size=(84, 84))
        return obs.squeeze(0).squeeze(0)

    def make_move(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            # lost a life? press FIRE to relaunch the ball before continuing
            current_lives = info.get("lives", self.lives)
            if current_lives < self.lives and not terminated:
                obs, r, t, tr, info = self.env.step(1)  # FIRE
                total_reward += r
                terminated = terminated or t
                truncated  = truncated  or tr
            self.lives = current_lives

            if terminated or truncated:
                break

        self.frames.append(self.screen_tensor(obs))
        return self.stack_screen(), total_reward, terminated, truncated, info

    def reset(self):
        obs, info = self.env.reset()
        obs, _, _, _, _ = self.env.step(1)
        frame = self.screen_tensor(obs)
        for _ in range(4):
            self.frames.append(frame)
        return self.stack_screen(), info
