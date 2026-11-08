# CS446 HW3: GridWorld Policy Evaluation and Improvement

This repo is Part 2 of CS446 Homework 3. It runs **iterative policy evaluation** and
**policy improvement** on a small GridWorld and prints the results for k = 1, 2, 3.

## The assignment

The environment is a 4×4 grid with one terminal state `T`. Cells use the assignment's numbering:

```
 1  2  3  4
 5  6  7  8
 9  T 10 11
12 13 14 15
```

- **Actions:** eight moves: N, NE, E, SE, S, SW, W, NW. If a move would leave the grid, the agent stays where it is.
- **Reward:** −1 on every transition.
- **Discount:** γ = 1 (undiscounted).
- **Policy evaluated:** uniform random, π(a|s) = 1/8 for each action.

For each k = 1, 2, 3 the program prints:

1. **V_k(s):** the value of every state after k synchronous sweeps of the Bellman
   expectation backup, starting from V₀ = 0.
2. **The improved (greedy) policy** from V_k, drawn as arrows. When several actions tie
   for best, the cell shows all of them. The terminal cell shows `T`.
3. **The q-value table** q_k(s, a) = −1 + V_k(s′) behind that policy. Greedy actions are
   marked with `*`.

Every sweep evaluates the uniform random policy. The greedy policy is printed at each step
but is not fed back into the next sweep. The math uses exact fractions (`fractions.Fraction`), so tied actions are found
by exact comparison. Values are rounded to 3 decimal places only when printed.

## Project layout

| File | Purpose |
|---|---|
| `gridworld.py` | The whole implementation, plus the `main()` driver |
| `tests/test_gridworld.py` | pytest suite (18 tests) |
| `output.txt` | Saved program output for k = 1, 2, 3 |
| `JOURNAL.md` | Development journal: design decisions, bugs found, open questions |
| `requirements.txt` | Test dependency (`pytest`) |
| `pytest.ini` | pytest config (lets the tests import `gridworld`) |
| `.claude/skills/` | Claude Code skills used during development. Not needed to run the code. |
| `skills-lock.json` | Lock file for those skills, also from development. Not needed to run the code. |

## Installation

You need **Python 3.12+**. The program uses only the standard library. pytest is needed
only to run the tests.

```bash
git clone <repo-url>
cd hw3
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

Print the results for k = 1, 2, 3:

```bash
python gridworld.py
```

Save them to a file:

```bash
python gridworld.py > output.txt
```

Run the tests:

```bash
python -m pytest -v
```

The policy grids use the Unicode arrows `↑ ↗ → ↘ ↓ ↙ ← ↖`. If your terminal or font shows
them as boxes, view `output.txt` in an editor with UTF-8 support.

## AI Usage
This software was co-developed using Claude Code on the command line, and every line of code was reviewed manually. The assignment specification was provided to the coding agent. I also used skills (provided in the repo) to ensure that the agent interviewed me for the details of the implementation and followed test-driven development principles (along with other best practices such as documenting itself with comments, DRY, and writing to a journal to document the process and any unresolved issues/bugs).
