---
name: improve-pr
description: Improve a PR by addressing review comments. Use when explicitly instructed to improve the PR
  or address comments on a PR. Ignore when reviewing a PR.
---

# Improve a pull request

First, pull the HEAD of the branch, and rebase on origin/main.

Address each comment one by one. Create a separate commit for each comment. Follow the project conventions from
`AGENTS.md`: all checks and tests must pass (`pdm check` and `pdm test`).

If the comment cannot be resolved, explain why in a reply to the comment.

If the comment has been addressed, summarize the resolution by replying to the comment. Do not mark the comment as
being resolved.

After addressing all comments, re-request a review by all commenters.
