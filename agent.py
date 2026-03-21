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
        self.snake = []             # current snake body (set in get_state)
        self.best_score = 0
        self.score_history = []
        if os.path.exists('model.pth'):
            print("Loaded saved model!")
            self.model.load_state_dict(torch.load('model.pth'))
        self.trainer = Trainer(self.model,lr=0.0001,gamma=0.9)        # the trainer

        
    def get_state(self, snake, food, direction):
        self.snake = snake  # needed by is_collision for body checks
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
        self.epsilon = max(10, 200 - self.n_games)
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
        

    def is_collision(self,snake,point):
        x, y = point
        # hit wall
        if x < 0 or x >= WIDTH // CELL or y < 0 or y >= HEIGHT // CELL:
            return True
        # hit body
        if point in snake:
            return True
        return False
    
    def get_vision(self, snake, food_list, direction):
        head = snake[0]
        x, y = head

        # 1. These are the 8 "Compass" directions (World coordinates)
        # Right, Left, Down, Up, and the 4 Diagonals
        # Index: 0:R, 1:L, 2:D, 3:U, 4:RD, 5:LD, 6:RU, 7:LU

        # 2. Re-map the lookup order so the radar is RELATIVE to the head
        # We want the first direction in 'lookup' to ALWAYS be 'Straight Ahead'
        if direction == (1, 0):    # Facing RIGHT
            lookup = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]
        elif direction == (-1, 0): # Facing LEFT
            lookup = [(-1,0), (-1,-1), (0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1)]
        elif direction == (0, -1): # Facing UP
            lookup = [(0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1)]
        else:                      # Facing DOWN
            lookup = [(0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1), (1,0), (1,1)]

        vision = []
        max_dist = WIDTH // CELL # Used to normalize 0.0 to 1.0

        # 3. Cast 8 rays in the relative directions
        for dx, dy in lookup:
            dist = 0
            found_food = 0
            found_body = 0

            # Start looking one step away from the head
            cx, cy = x + dx, y + dy

            while 0 <= cx < WIDTH // CELL and 0 <= cy < HEIGHT // CELL:
                dist += 1
                # Check if any food from our list is at this spot
                if (cx, cy) in food_list:
                    found_food = 1
                # Check if our own body is at this spot
                if (cx, cy) in snake:
                    found_body = 1

                # Move further along the ray
                cx += dx
                cy += dy

            # NORMALIZE: dist becomes a float between 0.0 and 1.0
            # This prevents "huge" numbers from confusing the Neural Network
            vision.extend([dist / max_dist, found_food, found_body])

        # 4. Add the 4 direction bits as a fallback (4 inputs)
        # Total inputs = (8 directions * 3 values) + 4 direction bits = 28
        dir_r = int(direction == (1, 0))
        dir_l = int(direction == (-1, 0))
        dir_d = int(direction == (0, 1))
        dir_u = int(direction == (0, -1))
        vision.extend([dir_r, dir_l, dir_d, dir_u])

        return vision
    
    
    def remember(self, state, action, reward, next_state, done):
        # store experience in memory
        self.memory.append((state, action, reward, next_state, done))

    def log_game(self, score):
        self.score_history.append(score)
        is_best = score > self.best_score
        if is_best:
            self.best_score = score
        avg = sum(self.score_history[-50:]) / len(self.score_history[-50:])
        print(
            f"Game: {self.n_games:4d} | Score: {int(score):3d} | Best: {int(self.best_score):3d} | "
            f"Avg(50): {avg:.1f} | Epsilon: {max(self.epsilon, 0):3d}"
            + (" *** NEW BEST ***" if is_best else "")
        )

    def train_long_memory(self):
        if len(self.memory) > 2000:
            sample = random.sample(self.memory, 1000)
        else:
            sample = list(self.memory)

        states, actions, rewards, next_states, dones = zip(*sample)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        torch.save(self.model.state_dict(), 'model.pth')
        if self.n_games % 10 == 0:
            print(f"  [replay loss: {loss:.4f}] model saved")

        