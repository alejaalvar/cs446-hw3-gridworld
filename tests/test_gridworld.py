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
