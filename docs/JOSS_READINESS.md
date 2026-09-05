# JOSS readiness for CesarAlex 1.0.0

This checklist reflects the JOSS requirements consulted on 5 September 2026.
The repository files are prepared as a first stable release candidate, but the
public development record and authors' declarations must be verified before
submission.

## Repository checklist

| JOSS item | Current preparation | Action before submission |
| --- | --- | --- |
| OSI-approved license | GPL-3.0-only in `LICENSE` | Confirm GitHub detects the license |
| Research application | Statement of need and published use documented | Keep links to the related article and dataset public |
| Installable software | PEP 517/621 `pyproject.toml` and `pip install -e .` | Test from a clean clone |
| Modular web application | Interface separated from a testable core library | Preserve this separation in future changes |
| Documentation | Portuguese and English READMEs plus usage, API, architecture, validation, and privacy guides | Ask a colleague unfamiliar with the project to follow them |
| Examples | Synthetic HTML and offline deterministic example | Run it from a clean clone |
| Tests and CI | Automated tests and Python 3.10–3.12 workflow | Confirm GitHub Actions is green |
| Community pathways | Contribution, support, and governance documents | Enable public issues and pull requests |
| Releases | Version files and changelog prepared for 1.0.0 | Create the `v1.0.0` tag and GitHub release after checks pass |
| Citation metadata | `CITATION.cff` prepared | Confirm GitHub renders “Cite this repository” |
| JOSS paper | `paper/paper.md` and `paper/paper.bib` prepared | Authors must review content and compile the PDF |
| AI disclosure | Explicit section in the paper | Each author must confirm its completeness and accuracy |
| Conflicts/funding | Draft statements included | Confirm with every author before submission |
| Archival DOI | Not required at initial submission | After review, archive the accepted release in Zenodo or Figshare and report its DOI |

## Screening gates that documentation cannot create

### Public development history

JOSS currently requires more than six months of public repository history with
active, iterative development. A recent upload containing many files is not a
substitute for distributed commits. Before submitting:

1. verify the date when the GitHub repository became public;
2. inspect whether meaningful commits span more than six months;
3. retain existing historical commits when integrating the 1.0.0 files;
4. use public issues and pull requests for remaining improvements;
5. do not create a new repository or rewrite history merely to make it look
   older.

If the public history does not yet meet six months, release 1.0.0 and continue
open development before submitting. This is the principal possible blocking
item.

### Evidence of research impact

The published Biblios article documents the research workflow that motivated
CesarAlex, including 16,203 Lattes curricula and approximately 1.1 million
researcher-publication records. The paper also cites the associated open OSF
dataset. In the repository, document any additional real uses, presentations,
external users, feature requests, or integrations. Evidence must be concrete;
future intentions alone are insufficient.

### Collaborative open-source practice

The software has three proposed authors, but public collaboration should also be
visible. Prefer issues for requirements and bugs, pull requests for substantial
changes, review comments for domain decisions, and release notes that acknowledge
contributors. Genuine user feedback should remain in its original public form.

## First stable release sequence

1. Have all three proposed authors approve `AUTHORS.md`, `CITATION.cff`, the
   affiliations, funding statement, conflict statement, and paper.
2. Add these files to the existing repository through ordinary commits or a
   reviewed pull request, preserving its history.
3. Ask a colleague who did not develop CesarAlex to install it from a clean
   clone, run the offline example, and test the interface.
4. Resolve any problems publicly and obtain green CI.
5. Compile and review the JOSS PDF produced by the draft workflow.
6. Tag the checked commit as `v1.0.0` and create the first GitHub release.
7. Confirm that the repository is public, clonable without registration, and
   permits public issues and proposed changes.
8. Confirm that the six-month public-history gate is satisfied.
9. Submit the repository and paper through the JOSS submission form, disclosing
   the related Biblios publication.
10. During review, respond publicly and promptly. After acceptance changes are
    complete, archive the approved release and provide its DOI.

## Information still requiring human confirmation

- all authors agree to JOSS co-authorship and accountability;
- contribution roles in `AUTHORS.md` are accurate;
- current affiliations in the paper and `CITATION.cff` are correct;
- CNPq process 308800/2022-2 and CAPES Finance Code 001 should be acknowledged
  for the software paper;
- there are no undisclosed conflicts of interest;
- the AI usage disclosure describes all tools used in the complete development
  history, not only the 1.0.0 refactoring;
- the existing public GitHub history satisfies the current screening gate.
