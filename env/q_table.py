from env.gridworld_env import ACTIONS
import random


class QLearningAgent:

    def __init__(self, env, actions = ACTIONS, alpha = 0.1, gamma = 0.8, epsilon = 0.5):
        self.env = env
        self.actions = actions
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = self.__init__q_table(env)

    def __init__q_table(self, env):
        """
        Initializes a Q-table for the environment.
        The Q-table starts with all zeros.
        """
        q_table = {}
        rows = len(env.grid)
        cols = len(env.grid[0])

        for r in range(rows):
            for c in range(cols):
                state = (r,c)
                q_table[state] = {a: 0.0 for a in self.actions}
        return q_table

    def choose_action(self, state):
        """
        Kitty chooses an action based on the current Q-values 
        in the current state.
        Employs epsilon-greedy strategy.
        """
        current_options = self.q_table[state]

        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            max_q = max(current_options.values())
            best_actions = [a for a, q in current_options.items() if q == max_q]
            return random.choice(best_actions)

    def update_q_value(self, state, action, reward, next_state):
        """
        Using the formula: 

        ---||  Q(s,a) <- Q(s,a) + Alpha*(R+gamma*max Q(s',a') - Q(s,a))  ||---

        The q-value for the current state given an action is updated once
        Kitty has taken the action and received a reward and the next state.
        """
        current_q = self.q_table[state][action]
        max_future_q = max(self.q_table[next_state].values())
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        
        self.q_table[state][action] = new_q
