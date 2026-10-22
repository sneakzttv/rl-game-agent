"""Small, inspectable reinforcement-learning environment; no external packages."""

from collections import deque
import math
import random


# Coordinates are (row, column), with row zero at the top of the browser board.
SIZE = 6
WALLS = frozenset({(1, 1), (1, 2), (2, 4), (3, 1), (4, 3)})
PITS = frozenset({(1, 4), (3, 3), (4, 1)})
GOAL = (5, 5)
START = (0, 0)
ACTIONS = ("up", "right", "down", "left")
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
MAX_STEPS = 60
STATES = tuple(
    (row, col)
    for row in range(SIZE)
    for col in range(SIZE)
    if (row, col) not in WALLS | PITS | {GOAL}
)


def transition(state, action):
    """Return next state, reward, terminal, event for one deterministic action."""
    if state not in STATES:
        raise ValueError("State must be a nonterminal, walkable cell.")
    if type(action) is not int or not 0 <= action < len(ACTIONS):
        raise ValueError("Action must be an integer from 0 to 3.")
    dr, dc = DELTAS[action]
    destination = (state[0] + dr, state[1] + dc)
    # A blocked action still costs a step, so bumping into walls is not free.
    if (
        not 0 <= destination[0] < SIZE
        or not 0 <= destination[1] < SIZE
        or destination in WALLS
    ):
        return state, -0.06, False, "wall"
    if destination in PITS:
        return destination, -1.0, True, "pit"
    if destination == GOAL:
        return destination, 1.0, True, "goal"
    return destination, -0.02, False, "move"


def greedy_action(table, state):
    """Stable tie-breaking makes evaluation and browser playback reproducible."""
    return max(range(4), key=lambda action: table[state][action])


def evaluate(table=None, *, episodes=200, seed=9001):
    """Evaluate a frozen greedy policy, or a uniform-random baseline.

    Starts are sampled from every safe nonterminal cell, not just the corner.
    Separate RNGs give both policies the same start distribution even though
    the random baseline consumes additional random numbers for its actions.
    """
    starts = random.Random(seed)
    actions = random.Random(seed + 1)
    wins = pits = timeouts = 0
    returns, lengths = [], []
    for _ in range(episodes):
        state = starts.choice(STATES)
        total = 0.0
        for step in range(1, MAX_STEPS + 1):
            action = actions.randrange(4) if table is None else greedy_action(table, state)
            state, reward, terminal, event = transition(state, action)
            total += reward
            if terminal:
                wins += event == "goal"
                pits += event == "pit"
                break
        else:
            timeouts += 1
        returns.append(total)
        lengths.append(step)
    return {
        "episodes": episodes,
        "seed": seed,
        "wins": wins,
        "pits": pits,
        "timeouts": timeouts,
        "success_rate": wins / episodes,
        "mean_return": sum(returns) / episodes,
        "mean_steps": sum(lengths) / episodes,
    }


def train(*, episodes=3000, seed=42, alpha=0.2, gamma=0.95):
    """Learn actual Q values with epsilon-greedy tabular Q-learning.

    Q(s,a) <- Q(s,a) + alpha * [reward + gamma*max Q(next,a) - Q(s,a)].
    Terminal transitions have no future value. Time limits truncate rollouts,
    not the underlying MDP, so the last nonterminal update still bootstraps.
    """
    if type(episodes) is not int or not 1 <= episodes <= 10000:
        raise ValueError("episodes must be an integer from 1 to 10000.")
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        raise ValueError("seed must be an integer from 0 to 2147483647.")
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be finite and in (0, 1].")
    if not math.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("gamma must be finite and in [0, 1).")
    rng = random.Random(seed)
    table = {state: [0.0] * 4 for state in STATES}
    history = []
    window_returns, window_wins = [], []
    for episode in range(1, episodes + 1):
        # A floor preserves exploration; random starts expose the whole board.
        epsilon = 0.03 + 0.47 * math.exp(-episode / max(episodes / 5, 1))
        state = rng.choice(STATES)
        total = 0.0
        won = False
        for step in range(1, MAX_STEPS + 1):
            action = (
                rng.randrange(4)
                if rng.random() < epsilon
                else greedy_action(table, state)
            )
            next_state, reward, terminal, event = transition(state, action)
            future = 0.0 if terminal else max(table[next_state])
            table[state][action] += alpha * (
                reward + gamma * future - table[state][action]
            )
            total += reward
            state = next_state
            if terminal:
                won = event == "goal"
                break
        window_returns.append(total)
        window_wins.append(won)
        # These are real observed training outcomes, not a fabricated curve.
        if episode % 50 == 0 or episode == episodes:
            history.append({
                "episode": episode,
                "epsilon": epsilon,
                "mean_return": sum(window_returns) / len(window_returns),
                "success_rate": sum(window_wins) / len(window_wins),
                "window_episodes": len(window_returns),
            })
            window_returns.clear()
            window_wins.clear()
    return table, history


def shortest_path(start):
    """A BFS reference is a test oracle, never the agent's policy."""
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        state, distance = queue.popleft()
        for action in range(4):
            nxt, _, terminal, event = transition(state, action)
            if event == "goal":
                return distance + 1
            if not terminal and nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, distance + 1))
    return None
