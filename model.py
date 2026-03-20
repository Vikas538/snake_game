import torch
import torch.nn as nn

class SnakeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(11, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 3)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # no relu on last layer
        return x




        
        # Step 3: calculate loss
        # Step 4: backpropagate
        # Step 5: update weights