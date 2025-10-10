import time
import numpy as np
import pygame
from env.gridworld_env import GridWorldEnv

def rollout_greedy(Q, level=0, fps=5):
    pygame.init()
    pygame.display.set_mode((1,1))  # init display for assets

    env = GridWorldEnv(
        level_files=["levels/level1.txt", "levels/level2.txt", "levels/level3.txt"],
        asset_dir="assets"
    )
    env.reset(level)
    screen = pygame.display.set_mode(env.get_window_size())
    clock = pygame.time.Clock()

    done = False
    while not done:
        s = env.get_state()
        a = int(np.argmax(Q[s]))  # always pick best move
        s2, r, done, _ = env.step(a)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        screen.fill((0,0,0))
        env.render_pygame(screen)
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()

if __name__ == "__main__":
    Q = np.load("Q.npy")
    rollout_greedy(Q, level=0)
