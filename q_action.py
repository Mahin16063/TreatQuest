from env.gridworld_env import GridWorldEnv
from agent.qagent import QAgent
import pygame
import numpy as np
import argparse


def run_visual_train(level=0, episodes=1000, alpha=0.8, gamma=0.75,
          eps_start=1.0, eps_end=0.05, eps_decay=800, delay=10):
    pygame.init()
    pygame.mixer.init()

    
    info = pygame.display.Info()
    screen_width = info.current_w
    screen_height = info.current_h
    pygame.display.set_mode((1, 1)) # Temporary Display

    # Initializing Environemnt #
    env = GridWorldEnv(
        level_files=["levels/level1.txt", "levels/level2.txt", "levels/level3.txt"],
        asset_dir="assets",
    )
    current_level = level
    env.reset(level)

    # Calculating Screen and Tile Size #
    grid_rows = len(env.grid)
    grid_cols = len(env.grid[0])
    margin = 100
    max_tile_width = (screen_width - margin) // grid_cols
    max_tile_height = (screen_height - margin) // grid_rows
    new_tile_size = min(max_tile_width, max_tile_height, 64)  # Maxium tile

    env.TILE_SIZE = new_tile_size
    env._load_assets()
    window_width = grid_cols * env.TILE_SIZE
    window_height = grid_rows * env.TILE_SIZE
    screen = pygame.display.set_mode((window_width, window_height))
    
    pygame.display.set_caption("TreatQuest: A Visual Training Demo")
    agent = QAgent(env.num_states, env.num_actions, alpha=alpha, gamma=gamma,
                   eps_start=eps_start, eps_end=eps_end, eps_decay_episodes=eps_decay)
    rewards = []
    for ep in range(episodes):
        env.reset(level)
        current_state = env.get_state()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 500:
            action = agent.select_action(current_state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

            if info["tile"] == "finished":
                if (level + 1) < len(env.level_files): # Moving to Next Level, Resetting Q-Table and Screen
                    level += 1
                    print(f"Level {level-1} completed. Next ;evel: {level}!")
                    env.reset(level)
                    current_state = env.get_state()
                    
                    screen = pygame.display.set_mode(env.get_window_size())
                    agent = QAgent(env.num_states, env.num_actions, alpha=alpha, gamma=gamma,
                                   eps_start=eps_start, eps_end=eps_end, eps_decay_episodes=eps_decay)
            
                else: # All Levels Are Completed
                    print("All levels completed!\nCongrats!")
                    done = True
            else: # Updating Q-Table With Next State and Rewards
                agent.update(current_state, action, reward, next_state, done) # Updating Q-table
                current_state = env.get_state()

            screen.fill((0, 0, 0))
            env.render_pygame(screen)
            env.render_ui(screen)
            pygame.display.flip()
            pygame.time.delay(delay) # Control Render Speed

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            agent.decay_epsilon(ep + 1)
            rewards.append(total_reward)
            if (ep + 1) % 50 == 0:
                print(f"Ep {ep+1:4d} | return={total_reward:6.2f} | eps={agent.epsilon:.2f}")

    pygame.quit()
    print("Training was Successful!\n Thank you for watching! \n- >^.^<")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--delay", type=int, default=100)
    args = parser.parse_args()
    run_visual_train(level=(args.level-1), episodes=args.episodes, delay=args.delay)