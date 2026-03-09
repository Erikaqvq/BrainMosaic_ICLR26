# Baselines (Public Release Status)

Baseline code is kept for reference only in this public release.

Current status:

- Data-dependent baseline scripts are intentionally **disabled** and will exit immediately.
- Private/internal dataset adapters are not released.
- You must provide your own dataset adapters and configs for internal reproduction.

Configuration guidance:

- Put all data/model paths in JSON under `baseline/configs/`.
- Do not hardcode machine-local paths in scripts.
