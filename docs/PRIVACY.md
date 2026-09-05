# Privacy and responsible use

Lattes curricula are public research records, but they can still contain
personal data. Public availability does not remove responsibilities related to
data minimization, research ethics, institutional policy, and applicable law.

## Data flow

- the uploaded HTML is parsed by the computer or server running Streamlit;
- ORCID values are sent to the public ORCID and OpenAlex endpoints used by the
  application;
- individual DOIs may be sent to OpenAlex when DOI checking is enabled;
- an OpenAlex key is sent to OpenAlex for authenticated requests;
- generated CSV files may contain names, identifiers, and publication metadata.

CesarAlex does not intentionally write API keys to its CSV reports. Users remain
responsible for the security of the execution environment and any server or
proxy logs outside the application.

## Recommended practices

- process only data for which you have a lawful and ethically appropriate use;
- prefer local execution for sensitive or large collections;
- do not commit API keys, downloaded curricula, or identifiable outputs to a
  public repository;
- use synthetic minimal examples in bug reports;
- remove unnecessary personal fields before sharing results;
- protect exported CSV files according to institutional policy;
- document collection dates and source limitations;
- interpret name-based author inference as a candidate match, not verified
  identity.

Researchers should obtain any institutional or ethics approval required for
their study. The software license grants rights to the code; it does not grant
rights to republish third-party personal data or source records.
