import pygame
import random
from agent import Agent

# --- SETUP ---
pygame.init()

WIDTH = 600
HEIGHT = 600
CELL = 30

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Self Playing Snake - Neural Network")

clock = pygame.time.Clock()
game_font = pygame.font.SysFont("Arial", 20)

# --- COLORS ---
BLACK        = (0, 0, 0)
GREEN        = (0, 200, 0)
BRIGHT_GREEN = (0, 255, 0)
RED          = (200, 0, 0)
WHITE        = (255, 255, 255)
GRAY         = (40, 40, 40)

# --- HELPERS ---
def random_food(snake):
    while True:
        x = random.randint(0, WIDTH // CELL - 1)
        y = random.randint(0, HEIGHT // CELL - 1)
        if (x, y) not in snake:
            return (x, y)

def is_collision(snake, point):
    x, y = point
    # wall check
    if x < 0 or x >= WIDTH // CELL or y < 0 or y >= HEIGHT // CELL:
        return True
    # body check
    if point in snake:
        return True
    return False

def action_to_direction(action, current_direction):
    # clockwise: RIGHT, DOWN, LEFT, UP
    clock_wise = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    idx = clock_wise.index(current_direction)

    if action == [1, 0, 0]:        # straight
        return clock_wise[idx]
    elif action == [0, 1, 0]:      # turn right
        return clock_wise[(idx + 1) % 4]
    else:                          # turn left
        return clock_wise[(idx - 1) % 4]

# --- INITIALIZE ---
snake     = [(5, 5), (4, 5), (3, 5)]
direction = (1, 0)
food      = random_food(snake)
score     = 0
lives     = 0
game_over = False
steps_without_food =0 
agent = Agent()

# --- MAIN LOOP ---
while True:

    # 1. HANDLE QUIT
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    if not game_over:

        # 2. GET CURRENT STATE (11 inputs)
        state = agent.get_state(snake, food, direction)

        # 3. GET ACTION ([1,0,0]=straight [0,1,0]=right [0,0,1]=left)
        action = agent.get_action(state)

        # 4. CONVERT ACTION TO DIRECTION
        direction = action_to_direction(action, direction)

        # 5. CALCULATE NEW HEAD
        new_head = (snake[0][0] + direction[0], snake[0][1] + direction[1])

        # 6. DEFAULT REWARD
        reward = 1  # survived one step

        # 7. CHECK COLLISIONS
        if is_collision(snake, new_head):
            reward    = -10
            game_over = True
        else:
            snake.insert(0, new_head)
            if new_head == food:
                reward = 10
                score += 1
                food = random_food(snake)
                steps_without_food=0
            else:
                steps_without_food+=1
                snake.pop()

        # 8. GET NEW STATE AFTER MOVE
        new_state = agent.get_state(snake, food, direction)

        # 9. TRAIN ON THIS STEP (short memory)
        max_steps = 50 * len(snake)
        if steps_without_food > max_steps:
            reward = -10
            game_over = True
        agent.train(state, action, reward, new_state, game_over)

    else:

        # 10. GAME OVER — train on all past memories (long memory)
        agent.train_long_memory()
        agent.n_games += 1
        agent.log_game(score)

        # reset
        lives    += 1
        score     = 0
        snake     = [(5, 5), (4, 5), (3, 5)]
        direction = (1, 0)
        food      = random_food(snake)
        game_over = False

        pygame.time.wait(300)

    # 11. DRAW
    screen.fill(BLACK)

    for x in range(0, WIDTH, CELL):
        pygame.draw.line(screen, GRAY, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL):
        pygame.draw.line(screen, GRAY, (0, y), (WIDTH, y))

    pygame.draw.rect(screen, RED, (food[0] * CELL, food[1] * CELL, CELL, CELL))

    for i, seg in enumerate(snake):
        color = BRIGHT_GREEN if i == 0 else GREEN
        pygame.draw.rect(screen, color, (seg[0] * CELL, seg[1] * CELL, CELL, CELL))
        pygame.draw.rect(screen, BLACK, (seg[0] * CELL, seg[1] * CELL, CELL, CELL), 1)

    info = game_font.render(f"Score: {score} | Game: {agent.n_games} | Epsilon: {agent.epsilon}", True, WHITE)
    screen.blit(info, (10, 10))

    pygame.display.flip()
    clock.tick(10)