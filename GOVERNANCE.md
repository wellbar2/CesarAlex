# Governance

## Project leadership

Wellington Barbosa Rodrigues is the lead maintainer and is responsible for
release management, technical decisions, and repository administration.
Rogério Mugnaini and Thamyres Vieira dos Santos provide domain direction and
validation for bibliometric and information-science requirements.

## Decision process

Bug fixes and small documentation improvements may be accepted by a maintainer
after automated checks pass. Changes that alter matching rules, output columns,
or the interpretation of coverage indicators should be discussed in a public
issue before implementation and reviewed by at least one domain contributor.

Decisions prioritize, in order:

1. correctness and traceability of bibliometric results;
2. compatibility with established user workflows and CSV outputs;
3. reduction of unnecessary external API requests;
4. maintainability and testability;
5. interface consistency.

When consensus is not reached, the lead maintainer records the decision and its
rationale in the relevant issue or pull request.

## Releases

The project follows semantic versioning. Every public release should have a Git
tag, release notes, an updated `CHANGELOG.md`, passing continuous integration,
and matching version numbers in `pyproject.toml`, `cesaralex/__init__.py`, and
`CITATION.cff`.

## Changes in governance

This document may be changed through a pull request. Maintainers may be added
after sustained contributions and agreement from the existing project team.
