#################### Importing Essential Libraries and APIs #######################

import os
os.environ["SDL_VIDEO_CENTERED"] = "1"
from env.gridworld_env import GridWorldEnv
from agent.qagent import QAgent
import pygame
import numpy as np
import argparse
import subprocess
import sys

#################### Level Background Music Functions #############################

LEVEL_MUSIC = {
    0: "assets/sounds/level_1.mp3",
    1: "assets/sounds/level_2.mp3",
    2: "assets/sounds/level_3.mp3",
    3: "assets/sounds/level_4.mp3",
}

def play_level_music(level_index: int, volume: float = 0.5):
    """Load and loop the background music for the given level index."""
    filename = LEVEL_MUSIC.get(level_index)
    if not filename:
        print(f"No music configured for level {level_index}")
        pygame.mixer.music.stop()
        return

    try:
        pygame.mixer.music.load(filename)
        pygame.mixer.music.set_volume(volume)
        # Loop forever with a small fade-in
        pygame.mixer.music.play(-1, fade_ms=800)
        print(f"Now playing music for level {level_index}: {filename}")
    except Exception as e:
        print(f"Error loading music for level {level_index}: {e}")
        pygame.mixer.music.stop()

########################### Function to Run Manual Mode #############################

def run_manual_play():
    """Run the main.py file for manual gameplay"""
    print("\n▶ Starting MANUAL PLAY...\n")
    pygame.quit()  # Close the current Pygame instance
    
    try:
        # Run main.py as a separate process
        subprocess.run([sys.executable, "main.py"])
    except FileNotFoundError:
        print("Error: main.py not found in the root directory!")
    except Exception as e:
        print(f"Error running main.py: {e}")
    
    # After manual play finishes, show the menu again
    main()


######################## Training Modes #####################################################

def train_by_completion(level=0, episodes=1000, alpha=0.9, gamma=0.9,
          eps_start=1.0, eps_end=0.05, eps_decay=800, delay=100):
    """
    Train the pet until the level is completed.
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
        level_files=["levels/level1.txt", "levels/level2.txt", "levels/level3.txt", "levels/level4.txt"],
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
    
    rewards = []
    episode = 0

    while level < len(env.level_files):
        env.reset(level)

        pygame.mixer.music.stop()
        play_level_music(level, volume=0.5)

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
            mode = "EXPLORE" if agent.epsilon > agent.eps_end else "EXPLOIT"
            env.render_hud(
                screen,
                mode=mode,
                episode=episode,
                total_reward=total_reward,
                epsilon=agent.epsilon
            )
            pygame.display.flip()
            pygame.time.delay(delay)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    np.save(f"q_table_level{level}.npy", agent.Q)
                    pygame.mixer.music.stop()
                    pygame.quit()
                    print(f"Episode {episode} on level {level} finished with total reward {total_reward}")
                    #agent.print_Q()
                    return

        # Completed Level #
        if info["tile"] == "finished":
            np.save(f"q_table_level{level}.npy", agent.Q)
            print(f"Level {level} completed! Moving to next level.")
            level += 1
            if level < len(env.level_files): # Move to Next Level
                #agent.print_Q()
                env.reset(level)
                current_state = env.get_state()
                
                screen = pygame.display.set_mode(env.get_window_size())
                agent = QAgent(env.num_states, env.num_actions, alpha=alpha, gamma=gamma,
                            eps_start=eps_start, eps_end=eps_end, eps_decay_episodes=eps_decay, env=env)
            else:
                print("All levels completed!\nCongrats!") # All Levels Done
                #agent.print_Q()
                pygame.mixer.music.stop()
                pygame.quit()
                return
        else:
            env.reset(level)
            current_state = env.get_state()

        agent.decay_epsilon(episode)
        rewards.append(total_reward)

    np.save(f"q_table_level{level}.npy", agent.Q)
    pygame.mixer.music.stop()
    pygame.quit()
    print("Training was Successful!\n Thank you for watching! >^.^<")


def train_by_episode(level=0, episodes=15, alpha=0.9, gamma=0.9, 
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
    
    for lev in range(level, len(env.level_files)):
        env.reset(lev)

        pygame.mixer.music.stop()
        play_level_music(level, volume=0.5)

        screen = pygame.display.set_mode(env.get_window_size())
        agent = QAgent(env.num_states, env.num_actions, alpha=alpha, gamma=gamma,
                        eps_start=eps_start, eps_end=eps_end, eps_decay_episodes=eps_decay, env=env)
        rewards = []
        for ep in range(episodes):
            env.reset(lev)
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
                mode = "EXPLORE" if agent.epsilon > agent.eps_end else "EXPLOIT"
                env.render_hud(
                    screen,
                    mode=mode,
                    episode=ep + 1,
                    total_reward=total_reward,
                    epsilon=agent.epsilon
                )
                pygame.display.flip()
                pygame.time.delay(delay)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.mixer.music.stop()
                        pygame.quit()
                        np.save(f"q_table_level{level}.npy", agent.Q)
                        print("Training interrupted. Q-table saved.")
                        #agent.print_Q()
                        return

            agent.decay_epsilon(ep + 1)
            rewards.append(total_reward)

            if (ep + 1) % 50 == 0:
                print(f"Ep {ep+1:4d} | return={total_reward:6.2f} | eps={agent.epsilon:.2f}")
        
        np.save(f"q_table_level{level}.npy", agent.Q)
        print(f"Training finished. Q-table saved for level {level}.")
    pygame.mixer.music.stop()
    pygame.quit()
    


def run_visual(level=0, delay=100):
    """
    Intakes a completed Q-table and chooses the best course of action
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
    
    for lev in range(level, len(env.level_files)):
        env.reset(lev)

        # new music for each level
        pygame.mixer.music.stop()
        play_level_music(lev, volume=0.5)

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

        try:
            q_table = np.load(f"q_table_level{lev}.npy")
        except FileNotFoundError:
            print(f"Missing q_table_level{level}.npy! Train first before running.")
            pygame.mixer.music.stop()
            pygame.quit()
            return
        
        current_state = env.get_state()
        done = False
        steps = 0

        while not done:
            action = np.argmax(q_table[current_state])
            next_state, reward, done, info = env.step(action)
            steps += 1

            screen.fill((0, 0, 0))
            env.render_pygame(screen)
            env.render_ui(screen)
            pygame.display.flip()
            pygame.time.delay(delay)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.mixer.music.stop()
                    pygame.quit()
                    return

            current_state = env.get_state()
    pygame.mixer.music.stop()
    pygame.quit()
    print("All levels completed! >^.^<")


################################## Menu Functions #####################################################
#easier way for us to run different modes of q_action.py from command line      

def show_menu():
    pygame.init()

    # ---------- CONFIG ----------
    WIDTH, HEIGHT = 1280, 720
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("TreatQuest - Main Menu")
    clock = pygame.time.Clock()

    # Play menu background music (loop = -1)
    try:
        pygame.mixer.music.load("menu_music.mp3")
        pygame.mixer.music.set_volume(0.5)   # 0.0 to 1.0
        pygame.mixer.music.play(-1, fade_ms=2000)   # Loop forever and fade in
    except Exception as e:
        print("Music load error:", e)


    # Load button click sound
    try:
        click_sound = pygame.mixer.Sound("click.wav")
        click_sound.set_volume(0.7)
    except Exception as e:
        print("Error loading click sound:", e)
        click_sound = None

    # ---------- LOAD BACKGROUND ----------
    try:
        bg = pygame.image.load("menu_bg2.png").convert()
        bg = pygame.transform.scale(bg, (WIDTH, HEIGHT))
    except Exception:
        # Fallback if background missing
        bg = pygame.Surface((WIDTH, HEIGHT))
        bg.fill((10, 10, 20))

    # ---------- LOAD TITLE ----------
    title_img = None
    title_rect = None
    try:
        title_img = pygame.image.load("treatquest_title.png").convert_alpha()
        # Scale down if too big
        max_title_width = WIDTH * 0.7
        if title_img.get_width() > max_title_width:
            scale_factor = max_title_width / title_img.get_width()
            new_size = (
                int(title_img.get_width() * scale_factor),
                int(title_img.get_height() * scale_factor),
            )
            title_img = pygame.transform.smoothscale(title_img, new_size)
        title_rect = title_img.get_rect(center=(WIDTH // 2, HEIGHT // 4))
    except Exception:
        # Fallback: render text if image not found
        font_title = pygame.font.SysFont(None, 80)
        title_img = font_title.render("", True, (255, 0, 0))
        title_rect = title_img.get_rect(center=(WIDTH // 2, HEIGHT // 4))

    # ---------- FONTS ----------
    font_button = pygame.font.SysFont(None, 42)

    # ---------- BUTTON DEFINITIONS ----------
    # (Label, value to return)
    button_defs = [
        ("Train by Completion", "1"),
        ("Train by Episode", "2"),
        ("Run Visual Mode", "3"),
        ("Manual Mode", "4"),
        ("Quit", "5"),
    ]

    buttons = []
    button_width = 380
    button_height = 70
    spacing = 20
    total_height = len(button_defs) * button_height + (len(button_defs) - 1) * spacing
    start_y = HEIGHT // 2 - total_height // 2 + 60  # shift a bit down

    for i, (label, value) in enumerate(button_defs):
        rect = pygame.Rect(
            WIDTH // 2 - button_width // 2,
            start_y + i * (button_height + spacing),
            button_width,
            button_height,
        )
        buttons.append((rect, label, value))

    # ---------- MAIN LOOP ----------
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return "5"  # Treat closing as Quit

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for rect, label, value in buttons:
                    if rect.collidepoint(mouse_pos):
                        # Play click sound
                        if click_sound:
                            click_sound.play()
                        pygame.mixer.music.fadeout(800)
                        pygame.mixer.music.stop()
                        pygame.quit()
                        return value


            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return "5"

        # ---- DRAW ----
        # Background
        screen.blit(bg, (0, 0))

        # Dark overlay to make UI readable
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        screen.blit(overlay, (0, 0))

        # Title
        screen.blit(title_img, title_rect.topleft)

        # Buttons
        for rect, label, value in buttons:
            hovered = rect.collidepoint(mouse_pos)

            # Semi-transparent "glass" button surface
            button_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)

            # Base alpha + extra when hovered
            base_alpha = 110
            hover_boost = 60 if hovered else 0
            alpha = max(0, min(255, base_alpha + hover_boost))

            # Fill with transparent dark color
            button_surf.fill((15, 15, 25, alpha))

            # Border: brighter on hover
            border_color = (255, 80, 80) if hovered else (220, 220, 240)
            pygame.draw.rect(button_surf, border_color, button_surf.get_rect(), 3)

            # Blit button panel
            screen.blit(button_surf, rect.topleft)

            # Text
            text_surf = font_button.render(label, True, (255, 255, 255))
            text_rect = text_surf.get_rect(center=rect.center)
            screen.blit(text_surf, text_rect.topleft)

        pygame.display.flip()
        clock.tick(60)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--delay", type=int, default=50)
    args = parser.parse_args()

    choice = show_menu()

    if choice == "1":
        print("\n▶ Starting TRAIN BY COMPLETION...\n")
        train_by_completion(
            level=args.level,
            episodes=args.episodes,
            delay=args.delay
        )

    elif choice == "2":
        print("\n▶ Starting TRAIN BY EPISODE...\n")
        train_by_episode(
            level=args.level,
            episodes=args.episodes,
            delay=args.delay
        )

    elif choice == "3":
        print("\n▶ Starting VISUAL RUN...\n")
        run_visual(
            level=args.level,
            delay=args.delay
        )
    elif choice == "4":
        run_manual_play()

    else:
        print("\nExiting TreatQuest. Goodbye!\n")    


if __name__ == "__main__":
    main()