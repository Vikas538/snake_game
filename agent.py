import random
import numpy as np
from model import SnakeNet
from trainer import Trainer
from collections import deque
import torch
import os


WIDTH = 600
HEIGHT = 600
CELL = 30



class Agent:
    def __init__(self):
        self.n_games = 0        # how many games played
        self.epsilon = 0        # exploration rate
        self.gamma = 0.9        # discount factor
        self.memory = deque(maxlen=100_000)        # store past experiences
        self.model = SnakeNet()         # the neural network
        if os.path.exists('model.pth'):
            self.model.load_state_dict(torch.load('model.pth'))
            print("Loaded saved model!")
        self.trainer = Trainer(self.model,lr=0.001,gamma=0.9)        # the trainer

        
    def get_state(self, snake,food,direction):
        head = snake[0]
        x, y = head

        dir_r = direction == (1, 0)
        dir_l = direction == (-1, 0)
        dir_u = direction == (0, -1)
        dir_d = direction == (0, 1)
    
        state = [
            # danger straight
            (dir_r and self.is_collision((x+1, y))) or
            (dir_l and self.is_collision((x-1, y))) or
            (dir_u and self.is_collision((x, y-1))) or
            (dir_d and self.is_collision((x, y+1))),
    
            # danger right
            (dir_r and self.is_collision((x, y+1))) or
            (dir_l and self.is_collision((x, y-1))) or
            (dir_u and self.is_collision((x+1, y))) or
            (dir_d and self.is_collision((x-1, y))),
    
            # danger left
            (dir_r and self.is_collision((x, y-1))) or
            (dir_l and self.is_collision((x, y+1))) or
            (dir_u and self.is_collision((x-1, y))) or
            (dir_d and self.is_collision((x+1, y))),
    
            # current direction
            dir_r, dir_l, dir_u, dir_d,
    
            # food location
            food[0] < x,  # food left
            food[0] > x,  # food right
            food[1] < y,  # food up
            food[1] > y,  # food down
        ]
        
        return [int(s) for s in state]  # convert True/False to 1/0

        
    
    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        action = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)  # explore
            action[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()  # exploit
            action[move] = 1

        return action

    
    def train(self, state, action, reward, next_state, done):
        # train on one step
        self.remember(state, action, reward, next_state, done)
        self.trainer.train_step(state, action, reward, next_state, done)
        

    def is_collision(self, point):
        x, y = point
        # hit wall
        if x < 0 or x >= WIDTH // CELL or y < 0 or y >= HEIGHT // CELL:
            return True
        # hit body
        if point in self.snake:
            return True
        return False
    
    def remember(self, state, action, reward, next_state, done):
        # store experience in memory
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > 1000:
            sample = random.sample(self.memory, 1000)
        else:
            sample = list(self.memory)

        states, actions, rewards, next_states, dones = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        torch.save(self.model.state_dict(), 'model.pth')
        print(f"Model saved!")

        