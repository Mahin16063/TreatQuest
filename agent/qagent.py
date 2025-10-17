import numpy as np
import random
from env.gridworld_env import GridWorldEnv

class QAgent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay_episodes=800, env=None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_episodes = eps_decay_episodes
        self.epsilon = eps_start
        self.Q = np.zeros((num_states, num_actions))
        self.env = env

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, done):
        max_next = np.max(self.Q[next_state]) if not done else 0.0
        td_target = reward + self.gamma * max_next
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def decay_epsilon(self, episode):
        frac = min(episode / self.eps_decay_episodes, 1.0)
        self.epsilon = self.eps_start * (1 - frac) + self.eps_end * frac

    def print_Q(self):
        rows = len(self.env.grid)
        cols = len(self.env.grid[0])
        for r in range(rows):
            for c in range(cols):
                state = r * cols + c
                print(f"({r},{c}): {self.Q[state]}")
            print()