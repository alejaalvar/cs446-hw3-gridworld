"""Iterative policy evaluation and policy improvement on a 4x4 GridWorld.

CS446 HW3, Part 2. The MDP:

* States are grid cells ``(row, col)`` on a 4x4 grid; cell ``(2, 1)`` is terminal.
* Eight actions: N, NE, E, SE, S, SW, W, NW. A move that would leave the grid leaves
  the agent where it is.
* Reward is -1 on every transition; the problem is undiscounted (gamma = 1).
* The evaluated policy is uniform random: pi(a|s) = 1/8 for every action.

Run ``python gridworld.py`` to print V_k, the greedy (improved) policy, and the q-value
table for k = 1, 2, 3.
"""

from fractions import Fraction

State = tuple[int, int]
"""A grid cell as ``(row, col)``, with ``(0, 0)`` the top-left corner."""

ROWS = 4
"""Number of grid rows."""

COLS = 4
"""Number of grid columns."""

ACTIONS: dict[str, State] = {
    "N": (-1, 0),
    "NE": (-1, 1),
    "E": (0, 1),
    "SE": (1, 1),
    "S": (1, 0),
    "SW": (1, -1),
    "W": (0, -1),
    "NW": (-1, -1),
}
"""Action name -> ``(d_row, d_col)`` offset, in the spec's order N, NE, ..., NW."""

REWARD = Fraction(-1)
"""Reward received on every transition."""

GAMMA = Fraction(1)
"""Discount factor (the task is undiscounted)."""

TERMINAL: State = (2, 1)
"""The single terminal cell (shaded in the assignment figure)."""

STATES: list[State] = [(r, c) for r in range(ROWS) for c in range(COLS)]
"""Every cell in row-major order (top-left to bottom-right), terminal included."""

NON_TERMINAL_STATES: list[State] = [s for s in STATES if s != TERMINAL]
"""The 15 cells in which the agent acts."""

ValueFunction = dict[State, Fraction]
"""State-value estimates V(s); the terminal state always maps to 0."""

Policy = dict[State, dict[str, Fraction]]
"""A stochastic policy: state -> (action name -> probability pi(a|s))."""


def next_state(state: State, action: str) -> State:
    """Return the cell reached by taking ``action`` from ``state``.

    Args:
        state: The current cell ``(row, col)``.
        action: One of the keys of ``ACTIONS``.

    Returns:
        The resulting cell. If the move would leave the grid, the agent does not move
        and ``state`` is returned unchanged (the whole move is cancelled; coordinates
        are *not* clamped axis by axis).
    """
    d_row, d_col = ACTIONS[action]
    row, col = state[0] + d_row, state[1] + d_col
    if 0 <= row < ROWS and 0 <= col < COLS:
        return (row, col)
    return state


def q_values(values: ValueFunction, state: State) -> dict[str, Fraction]:
    """One-step lookahead action values from ``state`` under ``values``.

    Because transitions are deterministic, q(s, a) = REWARD + GAMMA * V(s'), where
    s' = ``next_state(s, a)``. Shared by policy evaluation, policy improvement, and the
    printed q-table, so the Bellman backup lives in exactly one place.

    Args:
        values: Current value estimates V_k for every state (terminal = 0).
        state: A non-terminal cell.

    Returns:
        Action name -> q-value, in ``ACTIONS`` order.
    """
    return {a: REWARD + GAMMA * values[next_state(state, a)] for a in ACTIONS}


def greedy_actions(values: ValueFunction) -> dict[State, tuple[str, ...]]:
    """Policy improvement: every action that maximizes q(s, a) in each state.

    Ties are kept (the spec asks for all equally good actions); because values are exact
    ``Fraction``s, a tie is plain equality with no float tolerance.

    Args:
        values: Value estimates V_k for every state.

    Returns:
        State -> tuple of greedy action names in ``ACTIONS`` order. The terminal
        state maps to an empty tuple because no action is taken there.
    """
    greedy = {}
    for state in values:
        if state == TERMINAL:
            greedy[state] = ()
            continue
        q = q_values(values, state)
        best = max(q.values())
        greedy[state] = tuple(a for a, v in q.items() if v == best)
    return greedy


def uniform_random_policy() -> Policy:
    """The equiprobable policy pi(a|s) = 1/8 for every action in every non-terminal state.

    Returns:
        A ``Policy`` covering ``NON_TERMINAL_STATES``.
    """
    prob = Fraction(1, len(ACTIONS))
    return {s: {a: prob for a in ACTIONS} for s in NON_TERMINAL_STATES}


def evaluate_policy(policy: Policy, k: int) -> list[ValueFunction]:
    """Iterative policy evaluation for ``k`` synchronous sweeps, starting from V_0 = 0.

    Args:
        policy: The policy to evaluate (action probabilities per non-terminal state).
        k: Number of sweeps to run.

    Returns:
        ``[V_0, V_1, ..., V_k]``.
    """
    v0: ValueFunction = {s: Fraction(0) for s in STATES}
    return [v0]
