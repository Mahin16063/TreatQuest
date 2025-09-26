from env.gridworld_env import ACTIONS
import random


def __init__q_table(env):
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
            q_table[state] = {a: 0.0 for a in ACTIONS}
    return q_table

def choose_action(q_table, state, epsilon):
    """
    Kitty chooses an action based on the current Q-values 
    in the current state.
    Employs epsilon-greedy strategy.
    """
    current_options = q_table[state]

    if random.random() < epsilon:
        return random.choice(ACTIONS)
    else:
        max_q = max(current_options.values())
        best_actions = [a for a, q in current_options.items() if q == max_q]
        return random.choice(best_actions)

def update_q_value(q_table, state, action, reward, next_state, alpha, gamma):