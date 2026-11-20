# Repository instructions for AI coding tools

Read README.md, docs/ARCHITECTURE.md, docs/STATUS.md, and CONTRIBUTING.md before editing.
The current app uses calibration/, detection.py, ball_wasb.py, and match_processor.py.
The legacy run.py / pipeline.py path and pythonProject1/ are reference implementations.

## Human commit accountability

- Do not create commits, amend commits, merge, or push. The human contributor reviews the
  diff and performs those operations using their own account and Git identity.
- Never set or override Git user.name, user.email, GIT_AUTHOR_*, GIT_COMMITTER_*, or --author.
  Do not impersonate another contributor or use an AI/bot identity.
- Do not add AI Co-Authored-By, Signed-off-by, or other AI authorship trailers.
- Do not remove legitimate human attribution. Do not rewrite existing commit history.
- AI assistance may be described honestly in PRs or course disclosures as required; human
  authorship does not mean the code was written without tools.
- Leave a concise handoff describing changes, checks run, and remaining limitations.

## Engineering conventions

- Follow docs/CODING_STANDARDS.md. Run make check after code changes.
- Add a regression test for behavior fixes. Keep routine tests independent of private clips,
  API keys, external model repositories, and weight downloads.
- For calibration/detection changes, inspect rendered overlays as well as numeric tests.
  Do not report historical benchmark results as newly verified measurements.
- Use original video frame indices; explicitly distinguish raw and sampled frames.
- Pitch coordinates are corner-origin metres: x in [0, length], y in [0, width].
- Preserve invalid calibration as unavailable; never fill it just to improve coverage metrics.
- Preserve existing user changes, local footage, weights, and experimental scratch files.
- Never log secrets, commit .env, or add generated videos/weights/datasets to Git.
- Do not change repository settings or invite contributors unless the user authorizes it.
