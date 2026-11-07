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

from decimal import ROUND_HALF_UP, Decimal
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

ARROWS: dict[str, str] = {
    "N": "↑", "NE": "↗", "E": "→", "SE": "↘",
    "S": "↓", "SW": "↙", "W": "←", "NW": "↖",
}
"""Action name -> arrow glyph used when printing a policy."""

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

STATE_LABELS: dict[State, str] = {s: str(i) for i, s in enumerate(NON_TERMINAL_STATES, 1)}
STATE_LABELS[TERMINAL] = "T"
"""Cell -> the state's number in the assignment figure (row-major, skipping T)."""

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

    Each sweep applies the Bellman expectation backup to every non-terminal state:

        V_{k+1}(s) = sum_a pi(a|s) * q_k(s, a),   q_k(s, a) = r + gamma * V_k(s')

    The update is *synchronous*: every new value is computed from the previous sweep's
    values only, never from values already updated in the current sweep. The terminal
    state stays at 0.

    Args:
        policy: The policy to evaluate (action probabilities per non-terminal state).
        k: Number of sweeps to run.

    Returns:
        ``[V_0, V_1, ..., V_k]``.
    """
    history: list[ValueFunction] = [{s: Fraction(0) for s in STATES}]
    for _ in range(k):
        prev = history[-1]
        new = dict(prev)  # copy, so reads below only ever see the previous sweep
        for state, action_probs in policy.items():
            q = q_values(prev, state)
            new[state] = sum(p * q[a] for a, p in action_probs.items())
        history.append(new)
    return history


def _format_value(value: Fraction) -> str:
    """Format an exact value to 3 decimal places, rounding halves away from zero.

    Python's float formatting rounds exact ties half-to-even (``-2.5625`` -> ``-2.562``),
    which disagrees with hand rounding (``-2.563``). Rounding the exact ``Fraction``
    through ``Decimal`` with ``ROUND_HALF_UP`` matches what is written on paper.

    Args:
        value: An exact value.

    Returns:
        The value as a string with exactly three decimals, e.g. ``"-1.875"``.
    """
    exact = Decimal(value.numerator) / Decimal(value.denominator)
    return str(exact.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def _format_table(rows: list[list[str]]) -> str:
    """Render rows of cell strings as a bordered, column-aligned text table.

    Shared by every renderer so all printed tables look the same.

    Args:
        rows: Table rows; every row must have the same number of cells.

    Returns:
        The table as a multi-line string, e.g. ``| a | b |`` lines between ``+--+`` rules.
    """
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    rule = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    lines = [rule]
    for row in rows:
        lines.append("| " + " | ".join(cell.rjust(w) for cell, w in zip(row, widths)) + " |")
        lines.append(rule)
    return "\n".join(lines)


def _grid_rows(cell_text) -> list[list[str]]:
    """Lay out one string per cell in the 4x4 GridWorld shape.

    Args:
        cell_text: Callable mapping a ``State`` to the text shown in that cell.

    Returns:
        ``ROWS`` lists of ``COLS`` strings, ready for ``_format_table``.
    """
    return [[cell_text((r, c)) for c in range(COLS)] for r in range(ROWS)]


def format_value_grid(values: ValueFunction) -> str:
    """Render V_k in the GridWorld layout, each value to 3 decimal places.

    Args:
        values: Value estimates for every state.

    Returns:
        A 4x4 text table.
    """
    return _format_table(_grid_rows(lambda s: _format_value(values[s])))


def format_policy_grid(greedy: dict[State, tuple[str, ...]]) -> str:
    """Render a greedy policy as arrows in the GridWorld layout.

    Each cell lists every tied greedy action as arrows in ``ACTIONS`` order; the
    terminal cell shows ``T``.

    Args:
        greedy: Output of ``greedy_actions``.

    Returns:
        A 4x4 text table.
    """
    return _format_table(
        _grid_rows(lambda s: "T" if s == TERMINAL else "".join(ARROWS[a] for a in greedy[s]))
    )


def format_q_table(values: ValueFunction) -> str:
    """Render the one-step lookahead q-values behind the improved policy.

    One row per non-terminal state (labelled with its PDF number), one column per action
    in ``ACTIONS`` order. Greedy entries, i.e. the arrows in ``format_policy_grid``, are
    marked with ``*``; other entries get a trailing space so decimals stay aligned.

    Args:
        values: The value estimates V_k used for policy improvement.

    Returns:
        A text table with a header row and 15 state rows.
    """
    greedy = greedy_actions(values)
    rows = [["s", *ACTIONS]]
    for state in NON_TERMINAL_STATES:
        q = q_values(values, state)
        rows.append([
            STATE_LABELS[state],
            *(_format_value(v) + ("*" if a in greedy[state] else " ") for a, v in q.items()),
        ])
    return _format_table(rows)
