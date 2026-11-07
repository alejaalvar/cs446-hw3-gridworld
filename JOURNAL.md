# Development Journal — HW3 Part 2 (GridWorld Policy Evaluation & Improvement)

This journal records how Part 2 was built with an AI coding agent (Claude Code), using
the `/grill-me` → `grilling` skill for design and the `/tdd` skill for implementation.
It is kept chronologically and ends with running lists of on-the-fly decisions,
unresolved problems, and areas worth revisiting.

## Problem summary

4×4 GridWorld, PDF numbering (T = terminal, at `(row, col) = (2, 1)`):

```
 1  2  3  4
 5  6  7  8
 9  T 10 11
12 13 14 15
```

8 actions (N, NE, E, SE, S, SW, W, NW); off-grid moves leave the state unchanged;
reward −1 per transition; γ = 1; uniform random policy π(a|s) = 1/8. Run iterative
policy evaluation and policy improvement for exactly k = 1, 2, 3.

## Design decisions (from the grilling interview)

| # | Decision | Why |
|---|---|---|
| Q1 | Every sweep evaluates the **uniform random policy**; the greedy policy from V_k is printed but **not fed back** into the next sweep. | The spec says "using the uniform random policy… evaluate the policy for three iterations", and the required outputs match Sutton & Barto Fig. 4.1. `evaluate_policy` takes the policy as a parameter, so switching to feedback (GPI) is a driver-only change. |
| Q2 | **Synchronous** sweeps: V_{k+1} is computed only from V_k. | Standard for hand calculation, so it matches Part 1. |
| Q3 | Pure standard-library Python 3.12; pytest is the only dependency. | 16 states do not need numpy. |
| Q4 | Exact arithmetic with `fractions.Fraction`. | Greedy ties become exact equality checks; no float-tolerance bugs. |
| Q5 | q(s,a) = −1 + V_k(s′), off-grid ⇒ s′ = s, V(T) = 0; keep **all** tied best actions; T has none. | Spec: "show all equally good actions". |
| Q6/Q12 | One script `gridworld.py`, tests in `tests/test_gridworld.py`. | ~150 lines; splitting would create shallow modules. |
| Q7 | This journal. | User request. |
| Q8 | Small commit on `main` after each green TDD cycle. | Makes the red→green process visible in git history. |
| Q9 | Program prints results only; the Part 1 comparison is written by the user. | Part 1 was done manually, outside this session. |
| Q10 | Values printed as decimals to 3 places. | Readable in the PDF; exact Fractions are used internally. |
| Q11 | Policy shown as Unicode arrows `↑ ↗ → ↘ ↓ ↙ ← ↖`, all ties per cell, `T` for terminal. | Spec prefers arrows. |
| Q13 | Constant `K = 3`, no CLI arguments. | Spec says exactly k = 1, 2, 3. |
| Q14 | Output to stdout only. | User redirects manually. |
| Q15/16 | A q-value table per k: rows = states 1–15, columns = N…NW, `*` marks greedy entries. | "Show your work" for policy improvement; helps trace any mismatch with Part 1. |
| Q17 | Per k: header → V_k grid → policy grid → q-table. | Mirrors the required output table. |
| Q18 | States are `(row, col)` tuples; PDF numbers are display labels only. | Movement is simplest as coordinate offsets. |
| Q19 | Test seams: `next_state`, `q_values`, `greedy_actions`, `evaluate_policy`, the three renderers, plus a `main()` smoke test. | Agreed before any test was written, per the `/tdd` skill. |

## Log

### 2026-11-06 — Setup

- Installed the skills (`grill-me`, `grilling`, `tdd`, `codebase-design`) into
  `.claude/skills/` and committed them with `skills-lock.json`. The initial install had
  put `grill-me` and `tdd` only under `.agents/skills/`, which Claude Code does not read,
  and the `grilling` and `codebase-design` skills they depend on were missing.
- Ran the grilling interview (decisions above).
- Installed `pytest==9.2.0` into `venv/`; recorded it in `requirements.txt`.
- Added `pytest.ini` with `pythonpath = .` so `tests/` can `import gridworld` without
  packaging the project.

### 2026-11-06 — Slice 1: `next_state` (actions + boundary rule)

- Red: interior move (6 –N→ 2) failed with `ModuleNotFoundError`; green after adding
  `ACTIONS` and offset arithmetic.
- Red: corner diagonal (1 –NW→ 1) failed because the move went to `(-1, -1)`; green after
  adding the in-grid check.
- **Caught a spec subtlety before it became a bug:** the plan said "clamp-to-grid", but
  clamping per axis is *wrong* for diagonals along an edge (NE from state 2 would clamp
  to state 3). The spec says off-grid moves leave the state unchanged, so the whole move
  is cancelled. Added a test pinning this.
- Added a test of all eight offsets from state 6. It and the edge test passed
  immediately (regression guards, not new behaviour).

### 2026-11-06 — Slices 2–3: `q_values` and `greedy_actions`

- `q_values`: red on state 6 under a hand-written V₁ (expected S = −1, others −2);
  green with a one-line dict comprehension over `ACTIONS`. This is the single place
  where the Bellman backup r + γV(s′) is written, so evaluation, improvement and the
  printed q-table all reuse it (DRY).
- `greedy_actions`: red/green for "neighbour of T moves into T" (state 6 → S), then
  red/green for "terminal has no actions" (added `TERMINAL`). The tie test (state 1
  keeps all 8 actions at k = 1) passed immediately because `max` + exact equality
  already keeps every tied action.

## On-the-fly decisions

- `pytest.ini` with `pythonpath = .` instead of a `conftest.py` path hack or a package
  layout — smallest change that lets tests import the top-level script.
- Boundary rule implemented as "cancel the whole move" rather than per-axis clamping
  (see Slice 1). Tests refer to states by PDF number via a hand-written `CELL` table so
  they read like the assignment.

## Unresolved problems

_None yet._

## Areas to revisit

- **Q1 interpretation.** If the instructor intends the improved policy to be fed back
  (generalized policy iteration), change the driver in `main()` to evaluate the greedy
  policy from the previous k instead of the uniform random policy. V₂ and V₃ will change.
