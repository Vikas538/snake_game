import pygame
import random
from agent import Agent

# --- SETTINGS ---
WIDTH, HEIGHT, CELL = 600, 600, 30
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Self-Playing Snake: Curriculum Learning")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 20)

# --- COLORS ---
BLACK        = (0, 0, 0)
WHITE        = (255, 255, 255)
RED          = (200, 0, 0)
GREEN        = (0, 200, 0)
HEAD_COLOR   = (0, 255, 100)
GRAY         = (40, 40, 40)

# --- HELPERS ---
def random_food(snake, existing_foods):
    while True:
        p = (random.randint(0, WIDTH // CELL - 1), random.randint(0, HEIGHT // CELL - 1))
        if p not in snake and p not in existing_foods:
            return p

def is_collision(snake, p):
    # Wall check
    if p[0] < 0 or p[0] >= WIDTH // CELL or p[1] < 0 or p[1] >= HEIGHT // CELL:
        return True
    # Body check
    if p in snake:
        return True
    return False

def action_to_direction(action, current_direction):
    # Clockwise order: Right, Down, Left, Up
    clock_wise = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    idx = clock_wise.index(current_direction)

    if action == [1, 0, 0]:    # Straight
        return clock_wise[idx]
    elif action == [0, 1, 0]:  # Turn Right
        return clock_wise[(idx + 1) % 4]
    else:                      # Turn Left
        return clock_wise[(idx - 1) % 4]

# --- INITIALIZE GAME ---
agent = Agent()
snake = [(5, 5), (4, 5), (3, 5)]
direction = (1, 0)
score = 0
steps_without_food = 0
game_over = False

# Create the permanent food list (starts with 40)
food_list = [random_food(snake, []) for _ in range(40)]

# --- MAIN LOOP ---
while True:

    # 1. HANDLE QUIT
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    if not game_over:
        reward = 0
        # 2. GET CURRENT STATE (28 inputs)
        state = agent.get_vision(snake, food_list, direction)

        # 3. GET ACTION
        action = agent.get_action(state)

        # 4. UPDATE DIRECTION
        direction = action_to_direction(action, direction)

        # 5. CALCULATE DISTANCE TO NEAREST FOOD (For Reward Shaping)
        head = snake[0]
        # Get distances to all remaining food pieces
        dists = [abs(head[0] - f[0]) + abs(head[1] - f[1]) for f in food_list]
        old_min_dist = min(dists) if dists else 0

        # 6. MOVE HEAD
        new_head = (head[0] + direction[0], head[1] + direction[1])

        # 7. CHECK COLLISIONS & EATING
        if is_collision(snake, new_head):
            reward =-10
            game_over = True
        else:
            reward+=0.0005
            snake.insert(0, new_head)
            if new_head in food_list:
                reward+= 10
                score += 1
                food_list.remove(new_head) # Food is gone forever
                steps_without_food = 0
                
                # Safety: if ALL food is gone, spawn 1 so the game doesn't break
                if len(food_list) == 0:
                    food_list.append(random_food(snake, []))
            else:
                # Check new distance to nearest food
                new_min_dist = min([abs(new_head[0] - f[0]) + abs(new_head[1] - f[1]) for f in food_list])
                
                # Small nudge: reward for getting closer, penalty for moving away
                reward+= 0.0005
                if new_min_dist < old_min_dist :
                    reward+=0.01
                else:
                    reward-=0.012
                
                steps_without_food += 1
                snake.pop()

        # 8. HUNGER LIMIT (Starvation)
        # We give it more time if food is scarce
        hunger_limit = 100 + (len(snake) * 10)
        if steps_without_food > hunger_limit:
            reward = -10
            game_over = True

        # 9. TRAIN SHORT MEMORY
        new_state = agent.get_vision(snake, food_list, direction)
        agent.train(state, action, reward, new_state, game_over)

    else:
        # 10. RESET FOR NEXT GAME
        agent.train_long_memory()
        agent.n_games += 1
        agent.log_game(score)

        # Reset snake and score, but KEEP the depleted food_list
        score = 0
        steps_without_food = 0
        snake = [(5, 5), (4, 5), (3, 5)]
        direction = (1, 0)
        game_over = False
        pygame.time.wait(200)

    # 11. DRAWING
    screen.fill(BLACK)

    # Draw Grid
    for x in range(0, WIDTH, CELL):
        pygame.draw.line(screen, GRAY, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL):
        pygame.draw.line(screen, GRAY, (0, y), (WIDTH, y))

    # Draw all food in the list
    for f in food_list:
        pygame.draw.rect(screen, RED, (f[0] * CELL, f[1] * CELL, CELL, CELL))

    # Draw Snake
    for i, seg in enumerate(snake):
        color = HEAD_COLOR if i == 0 else GREEN
        pygame.draw.rect(screen, color, (seg[0] * CELL, seg[1] * CELL, CELL, CELL))

    # Info display
    info = font.render(f"Food Left: {len(food_list)} | Game: {agent.n_games} | Score: {score} | reward :{reward}", True, WHITE)
    screen.blit(info, (10, 10))

    pygame.display.flip()
    clock.tick(30) # 30 FPS is good for watching it learn