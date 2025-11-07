import numpy as np
import random
from env.gridworld_env import GridWorldEnv

class QAgent:
    """
    QAgent implements a basic **Q-Learning** reinforcement learning algorithm.

    It maintains a Q-table (a 2D matrix of state-action values) and learns by
    iteratively updating this table based on experience.

    Each row of Q corresponds to a unique state (encoded as an integer).
    Each column corresponds to an available action (e.g., UP, DOWN, LEFT, RIGHT).

    The goal is for the agent to learn which actions maximize future rewards
    through repeated interaction with the environment.

    Typical training loop:
        1. Choose an action using epsilon-greedy policy (explore or exploit)
        2. Take that action in the environment
        3. Receive a reward and observe the next state
        4. Update Q-table using the Bellman equation
        5. Repeat for many episodes
    """
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.95, #initial alpha=0.1, gamma=0.95 #eps_decay_episodes=800
                 eps_start=1.0, eps_end=0.05, eps_decay_episodes=500, env=None):
        """
        Initialize the agent and its learning parameters.

        Args:
            num_states (int): total number of possible states in the environment, (rows * columns).
            num_actions (int): number of possible actions (e.g., 4 for UP/DOWN/LEFT/RIGHT).
            alpha (float): learning rate — how much new information overrides old.
                           Range [0,1]. Smaller = slower learning.
            gamma (float): discount factor — how much future rewards matter.
                           Range [0,1]. Larger = more long-term planning.
            eps_start (float): starting value of epsilon (exploration rate).
                               At start, the agent explores almost randomly.
            eps_end (float): final epsilon value (after full decay).
                             The lowest probability of random exploration.
            eps_decay_episodes (int): number of episodes to decay epsilon from start → end.
            env (GridWorldEnv): optional reference to environment (used for debugging/printing).
        """
        #environmental dimensions
        self.num_states = num_states #total states in grid
        self.num_actions = num_actions

        #learning hyperparameters
        self.alpha = alpha
        self.gamma = gamma

        #exploration/exploitation settings
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_episodes = eps_decay_episodes
        self.epsilon = eps_start

        #initialize the Q-table with zeros
        #each entry Q[s,a] represents the estimated value of taking action a in state s
        self.Q = np.zeros((num_states, num_actions))
        self.env = env

    def select_action(self, state):
        """
        Choose an action for the current state using the epsilon-greedy policy.

        - With probability ε (epsilon), choose a random action (exploration).
        - With probability (1 - ε), choose the action with the highest Q-value (exploitation).

        Args:
            state (int): the current encoded state (row * cols + col).

        Returns:
            action (int): index of the chosen action.
        """
        # Generate a random number between 0 and 1
        # If it's less than epsilon → explore (random action)
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            # Otherwise exploit: pick the action with the maximum Q-value for this state
            return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-value for a given (state, action) pair using the Bellman equation.

        Formula:
            Q(s, a) ← Q(s, a) + α [ r + γ * max_a' Q(s', a') - Q(s, a) ]

        Meaning:
            - Take current Q-value (old knowledge)
            - Add a correction term based on the difference (TD error)
              between the predicted and actual return

        Args:
            state (int): current state index before the action.
            action (int): chosen action index.
            reward (float): immediate reward from the environment.
            next_state (int): state index after taking the action.
            done (bool): True if episode ended (no future rewards expected).
        """
        # 1. Estimate best future value from next state
        # If episode finished, future reward = 0
        max_next = np.max(self.Q[next_state]) if not done else 0.0

        # 2. Compute target (expected total return)
        td_target = reward + self.gamma * max_next

        # 3. Temporal difference (TD) error — how far off our prediction was
        td_error = td_target - self.Q[state, action]

        # 4. Update Q-value with learning rate α
        self.Q[state, action] += self.alpha * td_error

    def decay_epsilon(self, episode):
        """
        Linearly decay epsilon from eps_start → eps_end across
        eps_decay_episodes number of episodes.

        Args:
            episode (int): current episode number.

        Behavior:
            - At episode 0 → ε = eps_start (100% exploration)
            - At episode = eps_decay_episodes → ε = eps_end (minimal exploration)
            - Between them → ε interpolates linearly
        """
        # Compute fraction of decay completed
        frac = min(episode / self.eps_decay_episodes, 1.0)

        # Linear interpolation from start → end
        self.epsilon = self.eps_start * (1 - frac) + self.eps_end * frac

    def print_Q(self):
        """
        Print the current Q-table in a grid format that matches the environment layout.
        Useful for debugging and visualizing learned values.
        """
        # Infer environment grid shape (requires env to be assigned)
        rows = len(self.env.grid)
        cols = len(self.env.grid[0])
        for r in range(rows):
            for c in range(cols):
                state = r * cols + c
                # Print each state's action values (one row per grid cell)
                print(f"({r},{c}): {self.Q[state]}")
            print() # blank line between rows