# Legal Contract Risk Analyzer

This repository contains an academic ML project for analyzing legal contracts across three stages:

1. Stage 1+2: extract clause spans from contracts and classify them by clause type using the CUAD dataset.
2. Stage 3: assess clause risk with an agent that uses retrieval and contract search.
3. Stage 4: generate a structured report with explanations and recommendations.

The detailed system design and target folder layout are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Current expectation for contributions

All new implementation work should follow the planned project structure.

- Stage 1+2 code belongs in `src/stage1_extract_classify/`
- Stage 3 code belongs in `src/stage3_risk_agent/`
- Stage 4 code belongs in `src/stage4_report_gen/`
- Shared utilities belong in `src/common/`
- Config files belong in `configs/`
- Tests belong in `tests/` and should mirror the `src/` structure
- One-off helper scripts belong in `scripts/`

Do not add stage implementation files to the repository root.

## Contribution process

Use this process for every future change:

Example: if you are fixing Stage 1+2 evaluation, first pull `main`, create a branch such as `fix/stage1-evaluation`, make only that change, test it, push the branch, and open a pull request that clearly says what was fixed and what is still not verified.

1. Pull the latest `main` branch.
2. Create a new branch for your work.
3. Make focused changes for one task or feature.
4. Commit with a clear message.
5. Push the branch to GitHub.
6. Open a pull request into `main`.
7. Wait for review before merging.

Example:

```bash
git checkout main
git pull origin main
git checkout -b feature/stage1-baseline-cleanup
# make changes
git add .
git commit -m "Move Stage 1 baseline code into src package"
git push origin feature/stage1-baseline-cleanup
```

## Branch naming

Use short, descriptive branch names.

- `feature/stage1-training`
- `feature/stage3-tools`
- `fix/stage1-evaluation`
- `docs/readme-update`

## What must be included when submitting code

Every pull request or code submission should include:

1. What stage the change belongs to.
2. What files were added, moved, or changed.
3. Why the change was needed.
4. What the code does.
5. How the code was tested.
6. What is still incomplete or not yet verified.
7. Any known issues, assumptions, or limitations.

A good submission note can be short, but it must be clear.

## Code expectations

Follow the existing repository rules in `.github/copilot-instructions.md`.

Key points:

- Use Python 3.10+
- Add type hints on functions
- Use `logging`, not `print()`, for status output
- Put model paths and hyperparameters in YAML config files under `configs/`
- Add docstrings for public functions
- Keep imports grouped as stdlib, third-party, local

## Testing expectations

Before submitting code:

- Add or update tests when the change affects behavior
- Keep tests under `tests/`
- Mirror the source layout in the test layout
- Use `pytest`
- Mock external API or LLM calls in tests

If something was not tested, say that clearly in the submission.

## What not to commit

Do not commit:

- large datasets
- model weights
- generated results
- secrets or API keys
- local virtual environments
- temporary notebooks or experiment outputs unless explicitly needed

## Review standard

A change is ready to merge when:

- the code is in the correct folder
- the purpose of the change is clear
- tests or validation are described
- known gaps are stated honestly
- another teammate has reviewed it

## Next cleanup for this repo

The current Stage 1+2 code now lives under `src/stage1_extract_classify/`. The next step is to review that code against the actual project requirements before treating Stage 1+2 as properly integrated.