# Contributing and GitHub Flow

This repository uses **GitHub Flow** (not the multi-branch GitFlow model).
The default branch is `main`. Changes are developed in short-lived branches,
reviewed via Pull Requests (PRs), checked by CI, and merged only after review.

## Branch and PR workflow

1. Update your local base branch:

   ```powershell
   git switch main
   git pull --ff-only origin main
   ```

2. Create a branch named for its purpose and optional Jira issue:

   ```powershell
   git switch -c feat/lspr-43-example
   ```

   Use `feat/` for features, `fix/` for fixes, `chore/` for tooling, and
   `docs/` for documentation. The current loader branch
   `feat/grid-raw-dataset-loader` follows this model.

3. Keep changes focused. Make meaningful commits, for example
   `test: validate GRID split samples` or `ci: run pytest on pull requests`.

4. Install and run tests before pushing:

   ```powershell
   python -m pip install -r requirements.txt
   python -m pip install pytest
   python -m pip install -e .
   python -m pytest tests/ -v -ra
   ```

5. Push your branch, then open a PR into `main`:

   ```powershell
   git push -u origin feat/lspr-43-example
   ```

6. In the PR, link the Jira task, summarize the change, record test evidence,
   address review feedback, and wait for GitHub Actions to pass.
   Do not merge a draft PR, a failing PR, or code that has not been reviewed.
   After approval, merge and remove the obsolete feature branch.

## CI: what runs automatically?

The workflow at `.github/workflows/python-tests.yml` runs on PRs into
`main`, pushes to `main`, and manual dispatch.

- Runner: Ubuntu / Python 3.11, with CPU-only PyTorch pre-installed.
- Installs repository dependencies, pytest, and the editable package.
- Runs `python -m pytest tests/ -v -ra`.
- Does not download GRID, mount Google Drive, generate media, or require secrets.
- Real GRID integration tests **skip** when `LIPSYNC_DATA_ROOT` is not set.
  A green CI status therefore confirms local-data-independent tests only,
  not real dataset validation.

### Verifying the real GRID dataset (maintainer machine)

Configure `LIPSYNC_DATA_ROOT` to point to a locally accessible copy of the
GRID data. Never commit this path or the dataset. Run:

```powershell
python -m pytest tests/datasets/test_manifest_integration.py -v -ra
python -m pytest tests/datasets/test_grid_dataset_integration.py -v -ra
python -m pytest tests/ -v -ra
```

The dataset integration test files above are part of the loader feature branch
(PR #9); they are not present in the earlier `main` skeleton until that PR
merges. Report the machine, Python version, result summary, and any skips in
the PR instead of treating an unrun test as passed.

## Dataset Loader delivery — LSPR-43

- [ ] PR #9 has been updated with the latest Loader changes and real-data test
      evidence, including the integration test for train/val/test.
- [ ] Automated CI is green on the current PR head commit.
- [ ] Review the loader implementation, error handling, file handle cleanup,
      security checks, and source/docs/dependency consistency.
- [ ] Explicitly track the audio/video duration gap as follow-up temporal
      alignment work (not a loader merge blocker by itself).
- [ ] Merge PR #9 into `main` after approvals and confirm the post-merge CI.

Note: repository branch protection (required status checks and PR approval
rules) must be configured in GitHub Settings by a maintainer; adding this
workflow alone does **not** enforce a merge gate.
