---
title: 'CesarAlex: Integrating Lattes curricula, ORCID, and OpenAlex for bibliometric coverage analysis'
tags:
  - Python
  - bibliometrics
  - scientometrics
  - Lattes Platform
  - ORCID
  - OpenAlex
authors:
  - name: Wellington Barbosa Rodrigues
    orcid: 0000-0002-6465-6012
    corresponding: true
    affiliation: "1, 2"
  - name: Thamyres Vieira dos Santos
    orcid: 0000-0003-3644-8600
    affiliation: "2, 3"
  - name: Rogério Mugnaini
    orcid: 0000-0001-9334-3448
    affiliation: 2
affiliations:
  - name: Universidade Federal do ABC, Brazil
    index: 1
  - name: Universidade de São Paulo, Brazil
    index: 2
  - name: Hospital Israelita Albert Einstein, Brazil
    index: 3
date: 5 September 2026
bibliography: paper.bib
---

# Summary

CesarAlex is open-source research software that compares publication records
from three sources: Brazilian Lattes curricula, ORCID profiles, and OpenAlex.
It reads curricula exported as HTML, normalizes Digital Object Identifiers
(DOIs), retrieves public publication metadata, and creates an auditable table
showing where each researcher-publication pair was found. Researchers can use a
web interface to process multiple curricula, download individual and
consolidated CSV reports, and inspect flow and source-overlap visualizations.
The software also monitors OpenAlex usage and minimizes avoidable requests by
caching DOI resolutions and conservatively inferred author profiles.

# Statement of need

The Lattes Platform is central to documenting scientific activity in Brazil,
but its records coexist with international identifier infrastructures. ORCID
connects researchers to self-managed records, while OpenAlex provides an open
scholarly knowledge graph [@haak2012orcid; @priem2022openalex]. Coverage and
metadata differ across these sources, making manual comparison slow,
irreproducible, and susceptible to normalization errors. The problem is
especially relevant to bibliometric studies, institutional data curation, and
research-management workflows that need to distinguish missing metadata from
genuinely absent publications.

CesarAlex provides a reproducible, researcher-centered integration workflow for
bibliometricians, information scientists, librarians, and research managers.
For each DOI it records source presence, available ISSN-L, candidate OpenAlex
author identities, the matching route, and a matching score. It can also retain
curricula with no ORCID and, when explicitly requested, investigate their DOIs
through OpenAlex. Outputs use stable, human-readable CSV conventions suited to
inspection and further analysis.

# State of the field

`scriptLattes` extracts and summarizes group-level knowledge from Lattes
curricula and established the value of open tooling for this national research
infrastructure [@menachalco2009scriptlattes]. PyAlex offers a lightweight Python
interface to the OpenAlex API [@pyalex], while `openalexR` supports general
OpenAlex data collection and bibliometric analysis in R
[@aria2023openalexr]. These tools address curriculum extraction or access to a
single bibliographic source. They do not directly implement CesarAlex's unit of
analysis: DOI-level, per-researcher coverage across Lattes, ORCID, and OpenAlex,
including no-ORCID recovery, conservative author inference, source-specific
presence flags, and ready-to-audit reports.

Building a focused integration layer was therefore preferable to extending a
general API client or a Lattes reporting system. CesarAlex uses the official
services underneath but encodes the cross-source decisions required by this
research design and exposes them through a workflow accessible without
programming.

# Software design

The Streamlit interface is separated from a testable Python domain core.
Immutable models represent publications, authors, curricula, and DOI-resolution
provenance; provider protocols decouple processing from HTTP clients. This lets
tests reproduce integration decisions without API credentials or network
availability.

DOI was selected as the principal equality key because it can be normalized
deterministically across all three sources. This improves precision and
traceability but excludes works without DOI from the central comparison. With a
declared ORCID, the processor retrieves ORCID and OpenAlex profiles and indexes
their publications locally. Without ORCID, optional singleton DOI lookup can
identify a unique strong author-name match. The associated OpenAlex profile is
then fetched once and searched before subsequent DOI calls. A cache miss returns
to the API and can discover an additional profile. Caches are limited to one
processing run to avoid stale cross-study state.

The application favors bounded, synchronous requests over aggressive
concurrency. This trade-off makes retries, multiple-key rotation, and usage
costs visible and predictable. Reports preserve the actual OpenAlex ID and name,
matching method, and score rather than converting heuristic inference into an
unqualified identity assertion.

# Research impact statement

The computational workflow from which CesarAlex evolved supported a published
study of 16,203 Lattes curricula and approximately 1.1 million
researcher-publication records [@rodrigues2026identifiers]. Its outputs enabled
the authors to measure overlap and complementarity among Lattes, ORCID, and
OpenAlex across Brazilian fields and regions; the resulting research data are
openly archived [@rodrigues2025dataset]. The study also exposed a methodological
gap: 4,879 curricula without ORCID could not enter the original automated
linkage. CesarAlex's optional DOI and inferred-profile path was developed to
investigate such cases while reducing repeated API calls. The current release
turns that research-specific workflow into modular, tested, documented software
that can be reused for other populations of Lattes curricula.

# AI usage disclosure

OpenAI Codex (GPT-5) assisted with refactoring for version 1.0.0, test
scaffolding, code review, and initial drafts of documentation and this paper.
The human authors defined the research problem, requirements, matching logic,
and design priorities; they reviewed, edited, and validated all assisted output
and remain responsible for the software and manuscript. Generative AI will not
be used to compose conversational responses to JOSS editors or reviewers,
except for translation if permitted by JOSS policy.

# Acknowledgements

The research context that motivated CesarAlex received support from CNPq
(productivity grant 308800/2022-2) and CAPES (Finance Code 001). The funders had
no role in software design decisions or preparation of this manuscript. The
authors declare no conflicts of interest.

# References
