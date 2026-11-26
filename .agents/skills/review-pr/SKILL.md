---
name: review-pr
description: Write a review for a pull request. Use when requested to review a pull request.
---

# Review Pull Request

Create a review in github, but just add comments. Do not approve or request changes. Do not modify or comment on other
pull requests or existing reviews.

Write a critical review, looking out for bugs, code structure, docstrings. Check for consistency, breaking changes,
dangling or incorrect references, etc. Check the modifications against the issue and PR description. In particular for
this codebase:

- New or changed functionality has tests in `tests/`, mirroring the package layout.
- The documentation in `docs/*.rst` is updated accordingly.
- Dependency changes are made through `pdm add`, with an updated `pdm.lock` lock file.

Number all findings for later reference. Write inline comments where possible.

Do not verify whether the checks or tests pass, as it is the responsibility of the owner.

Prefix the review with: "This review is AI-generated."

After the review is finished, create a new branch and address each finding in a separate commit. The branch name should
follow the scheme: `review/PR#`. Update the comments with references to the commits without backticks.
