# Pending envelopes

## Installation

Run `uv sync --extra dev` from the repository root.

## Usage

- Layout: `<run>/envelope.json` and `<run>/metadata.json`; metadata follows `published_result` in `schema.json`.
- Publishing happens on merge to main, through the publish-envelopes workflow.
- Envelopes must carry no hostname or network identifier.

Dry-run one pair locally:

```sh
uv run mlx-bench-publish-envelope \
  --pair published/pending/<run>/envelope.json published/pending/<run>/metadata.json \
  --dry-run
```
