from env.gridworld_env import ACTIONS


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

