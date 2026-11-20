# Contributing

## First setup

Follow [setup](docs/SETUP.md) and [personal key setup](docs/SECRETS.md). Install Gitleaks
for the secret-scanning hook, then configure your own Git identity:

```bash
git config user.name "Your Name"
git config user.email "your-verified-email@example.edu"
git config core.hooksPath .githooks
```

Use your own GitHub account and credentials; never share accounts. Each clone needs hook
setup. If you already use hooks, integrate this check instead of replacing your hook path.
The pre-commit hook rejects private config files and detected secrets. Gitleaks must be
installed before committing. The hook is a local guardrail and can be bypassed; CI repeats
the scan on every push.

## One task, one branch, one review

1. Create or claim an issue with owner, reviewer, dependencies, and acceptance criteria.
2. Move its board card from Backlog to Ready, then In progress. Use `feature/short-name`,
   `fix/short-name`, or `docs/short-name`.
3. Make a focused change. Add regression tests for behavior changes and update affected docs.
4. Run `make check`. For model/calibration changes, also inspect overlays on approved clips.
5. Review `git diff` and `git status`, then stage specific files and commit as yourself.
6. Open a PR referencing the issue, describe results/limits, and request the assigned reviewer.
7. Resolve feedback, merge through the PR workflow, and move the card to Done with evidence.

Use concise commit messages such as `fix: preserve video playback speed`. Do not commit
`.env`, weights, private clips, generated outputs, or unrelated local experiments. Preserve
third-party attribution and model provenance. Never rewrite shared history casually.

## Checks

```bash
make check
# Or, without Make:
python -m ruff check .
python -m ruff format --check .
python -m pytest
python -m compileall -q app.py run.py calibrate.py soccer_analyzer
```

Use `make format` for Python formatting and Prettier's defaults for files in `web/`. A local
`scratch/` folder is outside lint/test discovery; package code and root entry points are
checked. CI runs the model-free checks on Python 3.11 and 3.12 after pushes and PR updates.

For an integration/model evaluation, record clip identifier and permission, frame range,
resolution/FPS, settings, dependency/model versions, timing, coverage, observed errors, and
before/after overlays. Do not use calibration coverage alone as a measure of accuracy.

## Definition of done

Acceptance criteria met; checks pass; another team member reviews; docs match behavior;
evidence is linked from the issue; no secret or local artifact is staged. Research tasks may
finish with a reproducible negative result and a documented decision instead of a feature.
