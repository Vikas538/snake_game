import torch
import torch.nn as nn

class Trainer:
    def __init__(self, model, lr):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, done):
        # Step 1: convert state to tensor
        state = torch.tensor(state, dtype=torch.float)
        
        # Step 2: get prediction from model
        prediction = self.model(state)
      # forward pass
        target=prediction.clone()
        if done:
            target[action] = reward
        else:
            next_state = torch.tensor(next_state, dtype=torch.float)
            target[action] = reward + 0.9 * torch.max(self.model(next_state))
        
        loss = self.criterion(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()