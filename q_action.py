from env.gridworld_env import GridWorldEnv
from env.q_table import QLearningAgent, ACTIONS
import os
import pygame

NUM_EPISODES = 50

pygame.init()
pygame.display.set_caption("Treat Quest: Q-Learning Agent Demo")
screen = pygame.display.set_mode((1, 1))

level_files = ["levels/level1.txt", "levels/level2.txt", "levels/level3.txt"]
env = GridWorldEnv(level_files=level_files, asset_dir="assets")
agent = QLearningAgent(env, alpha=0.1, gamma=0.8, epsilon=0.5)

screen = pygame.display.set_mode(env.get_window_size())

for episode in range(NUM_EPISODES):
    env.reset(0)
    done = False
    reward = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        state = tuple(env.pet_pos)
        choose_action = agent.choose_action(state)
        _, tile = env.move_pet(choose_action)  # returns (level_changed, tile_type)

        if tile == "finished":
            reward = 10
            done = True
        elif tile == "trap":
            reward = -10
            done = True
        elif tile == "treat":
            reward = 10
        elif tile == "empty":
            reward = -1
        elif tile == "wall":
            reward = -2

        next_state = tuple(env.pet_pos)
        agent.update_q_value(state, choose_action, reward, next_state)

        env.render_pygame(screen)
        pygame.display.flip()
        pygame.time.delay(100)
        
    agent.epsilon *= 0.95

