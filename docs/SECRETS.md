# Personal API keys and secret handling

Each teammate runs the app on their own machine with their own Roboflow account/workspace
and private API key. Share model identifiers, source code and `.env.example`; never share
`.env`, credentials, authentication screenshots, or SDK cache directories.

## Get your own key

1. Sign in to your own [Roboflow account](https://app.roboflow.com/) and select/create your
   own workspace. Do not copy a teammate's key or use their account.
2. Open [API-key settings](https://app.roboflow.com/settings/api). Select **Generate New Key**
   if needed and copy the **private** API key locally. The publishable JavaScript key is not
   the key this Python app uses.
3. Verify access to the public model `football-players-detection-3zvbc/9` from your workspace.
   A private model requires separate access or a permitted copy; another person's key is
   not a substitute for model permissions.
4. Check your workspace's plan, credit usage and Flex billing settings. Do not assume local
   inference is free or that separate keys in one workspace create separate billing accounts.

Roboflow keys are workspace-scoped. Multiple/scoped keys in a shared workspace depend on
Roboflow's plan/features; the documented multiple-key option is an Enterprise feature. Use
separate personal workspaces for this local workflow rather than sharing one workspace key.
If a shared private model becomes necessary, agree on access and billing before changing this.
See [Roboflow's authentication documentation](https://docs.roboflow.com/reference/authentication/authentication/find-your-roboflow-api-key)
and [credit billing](https://docs.roboflow.com/platform/billing-and-plans/credits).

## Save it only on your machine

From the repository root, copy `.env.example` to `.env` **only if you do not already have a
local `.env`**. Open `.env` in your editor and paste your key after `ROBOFLOW_API_KEY=`. Keep
model paths local too. Avoid placing a literal secret in a terminal command because shell
history may retain it. Do not paste keys into chat, issues, PRs, logs, or commits.

On macOS/Linux, restrict the file to your OS account and check the ignore rule:

```bash
chmod 600 .env
git check-ignore .env
```

The second command should print `.env`. On Windows use your account's file permissions.
`git check-ignore` confirms an ignore rule, not encryption; never use `git add -f .env`.
The tracked `.env.example` must contain an empty key. The app loads the repository-root `.env`;
an explicitly set `ROBOFLOW_API_KEY` environment variable takes precedence. If the key is
empty or unset, the optional Roboflow model is skipped, avoiding a fallback to another SDK key.
Restart Streamlit after changing a key so cached detector instances are replaced.

```bash
python -m streamlit run app.py
```

The startup summary indicates whether officials classification loaded; it never needs to
print your key. Without this model, COCO people detection and optional WASB ball detection
remain available. PnLCalib still needs its separate local weights. Tests require no API key.

## Keep keys out of commits

Install [Gitleaks](https://github.com/gitleaks/gitleaks) (CI uses 8.30.1): `brew install gitleaks`
on macOS, or use its official release/package instructions on other platforms. Enable the
repository hooks in every clone:

```bash
git config core.hooksPath .githooks
python3 scripts/check_secrets.py --staged
```

The pre-commit hook checks human identity, rejects private configuration files in the index,
and runs Gitleaks with redacted output. It fails if Gitleaks is missing. GitHub Actions also
scans fetched history. Local hooks can be bypassed and scanners can miss secrets; review the
staged diff yourself. Never attach an unredacted scanner report to an issue or PR.

## If a key is exposed

1. In Roboflow's API-key settings, **Roll API Key** invalidates the previous value and creates
   a replacement. If the dashboard instead offers revoke/create controls, revoke the old key
   and create a replacement. Update only local `.env` or the deployment secret store.
2. Check workspace usage and billing for unexpected activity. Restart affected applications.
3. Remove the secret from current files. If it entered Git, coordinate a history rewrite with
   the repository maintainer. Use a fresh clone after cleanup; do not merge old history back.
4. Existing downloaded copies and provider-side cached Git objects cannot be erased merely by
   rewriting branches. Revocation is what makes the old value unusable. Follow
   [GitHub's sensitive-data cleanup guidance](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
   for retained PR/cached references.

## Shared hosting is a different deployment model

A process-wide `.env` on a shared server belongs to that server, not to each browser user.
Do not run this app as a shared service expecting per-user API-key isolation. The current
team workflow is one local process/environment per teammate. A shared deployment would need
authentication, a server-side secret store, explicit billing ownership, and per-user credential
handling before accepting personal keys. The app intentionally has no browser key-entry form.

## Optional OpenAI key (fast pipeline AI keyframes)

The fast keyframe pipeline can send one frame every ~2 s (plus a few calibration frames) to an
OpenAI vision model. Use your own key in `.env` as `OPENAI_API_KEY=`; never paste it into chat,
issues, logs or commits. Every run has a hard spending cap (`--max-usd`, or "AI budget per clip"
in the app; measured cost was about $0.03-0.14 per 10 s clip). Frames leave your machine:
check consent before sending youth or private footage. Without a key the pipeline runs fully
offline (colour-clustered teams, no landmark fallback).
