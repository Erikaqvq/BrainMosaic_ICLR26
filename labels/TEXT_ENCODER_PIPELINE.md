# Text Encoder Asset Policy (Public Release)

This repository expects pre-built text assets and does not include the full preprocessing code.

Supported asset-design options for internal/private pipelines:

1. `word expansion (optional)`: expand each token into a short explanation phrase before embedding.
2. `fixed-dim truncation (optional)`: truncate embedding vectors to a fixed dimension.

Expected public artifacts consumed by training:

- `token_bank/embeddings.pt`
- `token_bank/map.json`
- `token_bank/info.json`
- `sentence_embeddings.pt`

You can choose either option (or both) when building private assets, as long as these artifact
formats remain consistent.
