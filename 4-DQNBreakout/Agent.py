import random 
import torch

class Agent():
    def __init__(self, model, num_actions, epsilon, device="cpu"):
        self.model = model
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.device = device

    def select_action(self, state):
        if random.random() < self.epsilon: # make a random action number 0,1,2,3
            return random.randint(0, self.num_actions -1)

        with torch.no_grad(): # turn off gradient graph
            q_values = self.model(state.to(self.device)) # returns the q values of each action (1,4)
        return q_values.argmax(dim=1).item() # return action number for largest q value

        