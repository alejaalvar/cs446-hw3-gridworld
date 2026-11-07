"""Behavioral tests for gridworld.py at the agreed seams.

Expected values are hand-derived literals (never recomputed the way the code does), and
states are referred to by their PDF numbers through the literal ``CELL`` table below.
"""

from fractions import Fraction

import gridworld as gw

# PDF state number -> (row, col), written out by hand from the assignment figure.
CELL = {
    1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (0, 3),
    5: (1, 0), 6: (1, 1), 7: (1, 2), 8: (1, 3),
    9: (2, 0), "T": (2, 1), 10: (2, 2), 11: (2, 3),
    12: (3, 0), 13: (3, 1), 14: (3, 2), 15: (3, 3),
}


# --- Seam 1: next_state (8 actions + boundary rule) -------------------------------

def test_interior_move_goes_to_neighbouring_cell():
    assert gw.next_state(CELL[6], "N") == CELL[2]


def test_diagonal_off_corner_leaves_state_unchanged():
    assert gw.next_state(CELL[1], "NW") == CELL[1]


def test_diagonal_off_edge_cancels_whole_move_rather_than_clamping():
    # NE from state 2 would clamp to state 3; the spec says the state is unchanged.
    assert gw.next_state(CELL[2], "NE") == CELL[2]


def test_all_eight_actions_from_state_6_reach_its_neighbours():
    reached = {a: gw.next_state(CELL[6], a) for a in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]}
    assert reached == {
        "N": CELL[2], "NE": CELL[3], "E": CELL[7], "SE": CELL[10],
        "S": CELL["T"], "SW": CELL[9], "W": CELL[5], "NW": CELL[1],
    }


# --- Seam 2: q_values (one-step lookahead) ---------------------------------------

# V_1 under the uniform random policy, written out by hand: -1 everywhere, 0 at T.
V1 = {cell: Fraction(-1) for label, cell in CELL.items() if label != "T"} | {CELL["T"]: Fraction(0)}


def test_q_values_reward_entering_terminal_over_other_moves():
    assert gw.q_values(V1, CELL[6]) == {
        "N": -2, "NE": -2, "E": -2, "SE": -2, "S": -1, "SW": -2, "W": -2, "NW": -2,
    }


# --- Seam 3: greedy_actions (policy improvement) ---------------------------------

def test_neighbour_of_terminal_greedily_moves_into_it():
    assert gw.greedy_actions(V1)[CELL[6]] == ("S",)


def test_terminal_state_has_no_greedy_actions():
    assert gw.greedy_actions(V1)[CELL["T"]] == ()


def test_all_tied_actions_are_kept_when_no_move_is_better():
    assert gw.greedy_actions(V1)[CELL[1]] == ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


# --- Seam 4: evaluate_policy (synchronous iterative policy evaluation) -----------

def test_evaluation_starts_from_zero_values():
    v0 = gw.evaluate_policy(gw.uniform_random_policy(), 0)[0]
    assert v0 == {cell: 0 for cell in CELL.values()}


def test_one_sweep_gives_minus_one_everywhere_except_terminal():
    assert gw.evaluate_policy(gw.uniform_random_policy(), 1)[1] == V1


def test_two_sweeps_favour_neighbours_of_terminal():
    # Hand-derived: each neighbour of T has exactly one action into T (q = -1) and seven
    # into -1 states (q = -2), so V_2 = -1 + (7 * -1 + 0) / 8 = -15/8; all others -2.
    near_t = Fraction(-15, 8)
    expected = {cell: Fraction(-2) for cell in CELL.values()} | {
        CELL[n]: near_t for n in (5, 6, 7, 9, 10, 12, 13, 14)
    } | {CELL["T"]: Fraction(0)}
    assert gw.evaluate_policy(gw.uniform_random_policy(), 2)[2] == expected


def test_three_sweeps_match_hand_calculation():
    v3 = gw.evaluate_policy(gw.uniform_random_policy(), 3)[3]
    assert (v3[CELL[1]], v3[CELL[6]]) == (Fraction(-95, 32), Fraction(-43, 16))


# --- Seam 5: renderers ------------------------------------------------------------

def table_rows(text):
    """Split a rendered ``| a | b |`` table into lists of stripped cell strings."""
    return [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in text.splitlines()
        if line.strip().startswith("|")
    ]


def test_value_grid_shows_values_to_three_decimals_in_grid_layout():
    v2 = gw.evaluate_policy(gw.uniform_random_policy(), 2)[2]
    assert table_rows(gw.format_value_grid(v2)) == [
        ["-2.000", "-2.000", "-2.000", "-2.000"],
        ["-1.875", "-1.875", "-1.875", "-2.000"],
        ["-1.875", "0.000", "-1.875", "-2.000"],
        ["-1.875", "-1.875", "-1.875", "-2.000"],
    ]


def test_value_grid_rounds_exact_halves_away_from_zero_like_hand_rounding():
    values = {cell: Fraction(-41, 16) for cell in CELL.values()}  # -2.5625 exactly
    assert table_rows(gw.format_value_grid(values))[0][0] == "-2.563"
