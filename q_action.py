import os
os.environ["SDL_VIDEO_CENTERED"] = "1"
from env.gridworld_env import GridWorldEnv
from agent.qagent import QAgent
import pygame
import numpy as np
import argparse


def train_by_completion(level=0, episodes=1000, alpha=0.9, gamma=0.9,
          eps_start=1.0, eps_end=0.05, eps_decay=800, delay=100):
    """
    Train the pet until the level is completed.
    Save the q_table for each level at the end of training.
    """
    pygame.init()
    pygame.mixer.init()

    #loads the bg music, loops forever #
    pygame.mixer.music.load("assets/sounds/background_music.mp3")
    pygame.mixer.music.play(-1)
    
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
                   eps_start=eps_start, eps_end=eps_end, eps_decay_episodes=eps_decay, env=env) # Initializing Agent
    rewards = []
    episode = 0
    while level < len(env.level_files):
        env.reset(level)
        current_state = env.get_state()
        done = False
        total_reward = 0
        steps = 0
        episode += 1

        while not done and steps < 500:
            action = agent.select_action(current_state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            agent.update(current_state, action, reward, next_state, done)
            current_state = next_state

            screen.fill((0, 0, 0))
            env.render_pygame(screen)
            env.render_ui(screen)
            pygame.display.flip()
            pygame.time.delay(delay)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    np.save(f"q_table_level{level}.npy", agent.Q)
                    print(f"Episode {episode} on level {level} finished with total reward {total_reward}")
                    agent.print_Q()
                    return

        # Completed Level #
        if info["tile"] == "finished":
            print(f"Level {level} completed! Moving to next level.")
            level += 1
            if level < len(env.level_files): # Move to Next Level
                env.reset(level)
                current_state = env.get_state()
                
                screen = pygame.display.set_mode(env.get_window_size())
                agent = QAgent(env.num_states, env.num_actions, alpha=alpha, gamma=gamma,
                            eps_start=eps_start, eps_end=eps_end, eps_decay_episodes=eps_decay, env=env)
            else:
                print("All levels completed!\nCongrats!") # All Levels Done
                pygame.quit()
                return
        else:
            env.reset(level)
            current_state = env.get_state()

        agent.decay_epsilon(episode)
        rewards.append(total_reward)
    
    pygame.quit()
    print("Training was Successful!\n Thank you for watching! >^.^<")


def train_by_episode(level=0, episodes=50, alpha=0.9, gamma=0.9, 
                      eps_start=1.0, eps_end=0.05, eps_decay=800, delay=1):
    """
    Train the pet for a fixed number of episodes.
    Save the q_table for each level at the end of training.
    """
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
                   eps_start=eps_start, eps_end=eps_end, eps_decay_episodes=eps_decay, env=env)
    
    for level in range(len(env.level_files)):
        env.reset(level)
        screen = pygame.display.set_mode(env.get_window_size())
        agent = QAgent(env.num_states, env.num_actions, alpha=alpha, gamma=gamma,
                        eps_start=eps_start, eps_end=eps_end, eps_decay_episodes=eps_decay, env=env)
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
                agent.update(current_state, action, reward, next_state, done)
                current_state = next_state
                total_reward += reward
                steps += 1

                screen.fill((0, 0, 0))
                env.render_pygame(screen)
                env.render_ui(screen)
                pygame.display.flip()
                pygame.time.delay(delay)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        np.save(f"q_table_level{level}.npy", agent.Q)
                        print("Training interrupted. Q-table saved.")
                        return

            agent.decay_epsilon(ep + 1)
            rewards.append(total_reward)

            if (ep + 1) % 50 == 0:
                print(f"Ep {ep+1:4d} | return={total_reward:6.2f} | eps={agent.epsilon:.2f}")
        
        np.save(f"q_table_level{level}.npy", agent.Q)
        print(f"Training finished. Q-table saved for level {level}.")
    pygame.quit()
    
    

def run_visual(level=0, delay=100):
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
    new_tile_size = min(max_tile_width, max_tile_height, 64)  # Maximum tile

    env.TILE_SIZE = new_tile_size
    env._load_assets()
    window_width = grid_cols * env.TILE_SIZE
    window_height = grid_rows * env.TILE_SIZE
    screen = pygame.display.set_mode((window_width, window_height))
    
    pygame.display.set_caption("TreatQuest: A Visual Run")
    for level in range(len(env.level_files)):
        env.reset(level)
        current_state = env.get_state()
        done = False
        steps = 0
        try:
            level_file = np.load(f"q_table_level{level}.npy")
        except FileNotFoundError:
            print(f"Missing {f"q_table_level{level}.npy"}! Train first before running.")
            pygame.quit()
            return
        while not done:
                    action = np.argmax(level_file[current_state])
                    print(f"Best Action for state {current_state}: {action}")
                    next_state, reward, done, info = env.step(action)
                    steps += 1

                    screen.fill((0, 0, 0))
                    env.render_pygame(screen)
                    env.render_ui(screen)
                    pygame.display.flip()
                    pygame.time.delay(delay)

                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return

                    current_state = env.get_state()

    pygame.quit()
    print("All levels completed! >^.^<")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--delay", type=int, default=100)
    args = parser.parse_args()
    train_by_episode(level=args.level, episodes=args.episodes, delay=args.delay)
    file = np.load("q_table_level0.npy")
    # for state in range(file.shape[0]):
    #     print(f"State {state}: {file[state]}")
    