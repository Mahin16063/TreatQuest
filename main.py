import pygame
from env.gridworld_env import GridWorldEnv

def main():
    pygame.init()

    # Temporary tiny screen to allow convert_alpha() in image loading
    pygame.display.set_mode((1, 1))

    # Create environment
    env = GridWorldEnv(
        level_files=["levels/level1.txt", "levels/level2.txt", "levels/level3.txt"], asset_dir="assets"
    )
    env.reset(0)

    # Set the actual window size
    screen = pygame.display.set_mode(env.get_window_size())
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
        pygame.display.flip()
        clock.tick(10)

    pygame.quit()


if __name__ == "__main__":
    main()
