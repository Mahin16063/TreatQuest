import os

os.environ["SDL_VIDEO_CENTERED"] = "1"
import pygame
from env.gridworld_env import GridWorldEnv
from env.q_table import QLearningAgent


def main():
    pygame.init()

    # Get screen size
    info = pygame.display.Info()
    screen_width, screen_height = info.current_w, info.current_h

    # Temporary tiny screen to allow convert_alpha() in image loading
    pygame.display.set_mode((1, 1))

    # Create environment
    env = GridWorldEnv(
        level_files=["levels/level1.txt", "levels/level2.txt", "levels/level3.txt"],
        asset_dir="assets",
    )
    env.reset(0)

    # Calculate scaling factor
    grid_rows = len(env.grid)
    grid_cols = len(env.grid[0])
    margin = 100  # pixels to leave as margin

    # Initial tile size calculation
    max_tile_width = (screen_width - margin) // grid_cols
    max_tile_height = (screen_height - margin) // grid_rows
    new_tile_size = min(max_tile_width, max_tile_height, 64)  # limit max tile size

    # Ensure window fits on screen
    window_width = grid_cols * new_tile_size
    window_height = grid_rows * new_tile_size
    if window_width > screen_width or window_height > screen_height:
        # Recalculate tile size to fit exactly
        new_tile_size = min(
            (screen_width - margin) // grid_cols, (screen_height - margin) // grid_rows
        )
        window_width = grid_cols * new_tile_size
        window_height = grid_rows * new_tile_size

    # Set new tile size and reload assets
    env.TILE_SIZE = new_tile_size
    env._load_assets()  # reload images at new size

    print("Window size: ", (window_width, window_height))
    # Set the actual window size
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Treat Quest")

    clock = pygame.time.Clock()
    running = True

    while running:
        level_changed = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    level_changed = env.move_pet("UP")
                elif event.key == pygame.K_DOWN:
                    level_changed = env.move_pet("DOWN")
                elif event.key == pygame.K_LEFT:
                    level_changed = env.move_pet("LEFT")
                elif event.key == pygame.K_RIGHT:
                    level_changed = env.move_pet("RIGHT")

        # Immediately resize window if level changed
        if level_changed:
            screen = pygame.display.set_mode(env.get_window_size())

        screen.fill((0, 0, 0))
        env.render_pygame(screen)
        env.render_ui(screen)
        pygame.display.flip()
        clock.tick(10)

    pygame.quit()


if __name__ == "__main__":
    main()
