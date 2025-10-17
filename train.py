import argparse
import pygame
import matplotlib.pyplot as plt
from env.gridworld_env import GridWorldEnv
from agent.qagent import QAgent

def train(level=0, episodes=1000, alpha=0.1, gamma=0.95,
          eps_start=1.0, eps_end=0.05, eps_decay=800):

    # Tiny hidden window so pygame image loads don't crash
    pygame.init()
    pygame.display.set_mode((1,1))

    env = GridWorldEnv(
        level_files=["levels/level1.txt", "levels/level2.txt", "levels/level3.txt"],
        asset_dir="assets"
    )
    _ = env.reset(level)
    agent = QAgent(env.num_states, env.num_actions, alpha=alpha, gamma=gamma,
                   eps_start=eps_start, eps_end=eps_end, eps_decay_episodes=eps_decay)

    rewards = []
    for ep in range(episodes):
        env.reset(level)
        s = env.get_state()  # <-- get integer state index
        done, total, steps = False, 0.0, 0

        while not done and steps < 500:
            a = agent.select_action(s)
            s2, r, done, _ = env.step(a) # Results for new state
            agent.update(s, a, r, s2, done) # Updating Q-table
            s = s2
            total += r
            steps += 1

        agent.decay_epsilon(ep + 1)
        rewards.append(total)
        if (ep + 1) % 50 == 0:
            print(f"Ep {ep+1:4d} | return={total:6.2f} | eps={agent.epsilon:.2f}")

    plt.plot(rewards)
    plt.xlabel("Episode"); plt.ylabel("Return"); plt.title("Training Progress")
    plt.tight_layout(); plt.show()

    # Save Q-table for a visual demo later
    import numpy as _np
    _np.save("Q.npy", agent.Q)
    print("Saved Q-table to Q.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()
    train(level=args.level, episodes=args.episodes)
