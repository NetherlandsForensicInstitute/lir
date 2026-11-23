---
name: resolve-issue
description: Resolve an issue by drafting a pull request. Use when requested to create a pull request, or to fix or resolve a numbered issue. Ignore when addressing issues related to an existing pull request.
---

# Issue resolution

First assess the description for clarity and ambiguity. If the issue is specific enough, go ahead. If not, abort and
leave a comment to explain why.

If the issue can be addressed:

1. Fetch and check out the main branch
2. Create a new branch for the issue, with a name formatted as: TYPE/SCOPE/NUMBER (e.g. `fix/config/123`)
3. Create a plan to resolve the issue in one or more steps
4. Applying the necessary changes for each of the steps in a separate commit
5. Create a draft pull request to target branch `main`, with an adequate description

Follow the project conventions from `AGENTS.md` while working: use `pdm check` and `pdm test` to verify the changes,
use the `type: short description` commit message style, add tests for new functionality, and update the documentation
in `docs/` when functionality changes.

Build on existing branches, but NEVER modify existing branches or pull requests from another user. Do not request a
review or otherwise address other users, except as stated below.

## Create a pull request

In the pull request description for a fix/refactor, clearly state:

- what is the scope of the problem
- what is the current behavior
- what is the expected behavior
- what is the resolution proposed in the pull request
- what are the side effects, if any

In the description for other types of issues, clearly state:

- a summary of the change and why the change is necessary
- how the changes are implemented
- what are the side effects, if any

Prefix the PR description with: "This pull request is AI-generated and awaiting human revision."

Request a review from the user who opened the issue.
