# Coding standards

Use Python 3.11-compatible syntax, PEP 8 conventions, and Ruff's formatter with a preferred
100-character line length. Ruff's E4/E7/E9 and F rules catch import/style errors and likely
logic errors; E741 is excepted for established mathematical variable names. Long explanatory
strings may exceed 100 characters. `pyproject.toml` is the executable configuration.

- Four spaces; no tabs in Python. One statement per line; formatter-managed wrapping.
- `snake_case` functions/variables/modules; `PascalCase` classes; `UPPER_SNAKE_CASE` constants.
- Descriptive names at API boundaries; short mathematical names inside geometry calculations.
- Type annotations on new public functions; docstrings explaining units, shape, coordinate
  frame, return values, and significant errors. Explain reasoning in comments.
- Images are OpenCV BGR unless explicitly converted. Coordinates distinguish pixels and
  metres. Frame indices refer to original input frames, not sampled positions.
- Reject invalid public parameters early. Use context managers or `finally` for resources.
  Catch exceptions at deliberate fallback boundaries; surface actionable failures.
- Inject expensive services/detectors so tests can replace them. Keep I/O outside geometry
  calculations where practical. Avoid shared mutable per-clip state across users.
- Seed randomized algorithms for reproducible evaluation. Preserve unavailable/unknown states
  rather than inventing coordinates, teams, or successful outcomes.
- No credentials, absolute developer-machine paths, generated assets, or embedded weights.
- Update behavioral tests when behavior changes. Do not rewrite tests simply to accept a
  regression. Run visual checks for changes that affect calibration or detection output.

The maintained Python sources are formatted together. The historical `pythonProject1/`
prototype is frozen reference material and is excluded from the maintained lint scope.
