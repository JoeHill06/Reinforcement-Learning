import torch


class Network(torch.nn.Module):
    """Dueling DQN: shared CNN trunk, split Value / Advantage heads.

    Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))

    V learns "how good is this state" once; A only has to learn action
    preferences relative to the state's mean. In Breakout most frames have
    ~equivalent actions, so putting the heavy lifting in V is a big win.
    """

    def __init__(self, num_actions=4):
        super().__init__()
        self.num_actions = num_actions

        # Shared conv trunk — identical to the old architecture up to Flatten
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=8, stride=4),   # (B,4,84,84) -> (B,32,20,20)
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),  #             -> (B,64,9,9)
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),  #             -> (B,64,7,7)
            torch.nn.ReLU(),
            torch.nn.Flatten(),                                #             -> (B, 3136)
        )

        # Value head — one scalar V(s) per state
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(3136, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
        )

        # Advantage head — A(s, a) for each of the num_actions actions
        self.advantage_head = torch.nn.Sequential(
            torch.nn.Linear(3136, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_actions),
        )

    def forward(self, x):
        features  = self.features(x)              # (B, 3136)
        value     = self.value_head(features)     # (B, 1)
        advantage = self.advantage_head(features) # (B, num_actions)

        # Mean-subtraction (not max-subtraction) is the paper's recommended
        # combine rule: it removes the V/A identifiability ambiguity without
        # the scaling issues max introduces, and it trains more stably.
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q
