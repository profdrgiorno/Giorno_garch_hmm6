# Provenance, archive policy, and release constraints

- Keep the submitted manuscript and its underlying numerical results **versioned**. Do not silently overwrite earlier numbers after repairs.
- Commit custom source code and the complete **verified** public aggregate results. Raw full simulations and market inputs may require additional storage or permission. Large raw CSVs can be distributed as DOI-backed archives or Git LFS when authorized.
- `study_design_PUBLIC.json` is the **same scientific design** with only the original absolute local Windows output path changed to a relative illustrative path; label this as a sanitized release copy, not the exact byte-for-byte original JSON.
- Neither original market price series nor any confidential/proprietary trading indicator implementation is included. Nothing in this archive authorizes publishing those materials.
- Separate confirmed facts (DGP truth, observed code outputs) from simulations, empirical market forecasts and conjectural economic interpretations.
- Do not include private machine locations, usernames, passwords, access tokens or credentials in GitHub commits. `.gitignore` excludes research output directories by default.
- The research owner should choose an explicit software and data license before enabling third-party redistribution/reuse. Until then, see `LICENSE_NOTICE.md`.
- Missing raw full path outputs prevent independent recreation of all 500-path paired intervals from **only** these public aggregate tables. That is a limitation of this package, not a failure of the Monte Carlo already run.
