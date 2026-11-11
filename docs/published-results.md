# Published result contract

Canonical schema: [`schema.json`](../schema.json),
`definitions.published_result`. Hugging Face source URLs and the read date are
listed in [`huggingface-metadata-sources.md`](huggingface-metadata-sources.md).

## Hub model metadata

The runner records the model's 40-character Hub commit SHA in
`model_revision`. The publisher reads `ModelInfo` at that exact revision and
copies `model_id`, `pipeline_tag`, `library_name`, `card_data`, `safetensors`,
`gguf`, `config`, `gated`, and `tags` into these fields:

`model_id`, `model_revision`, `pipeline_tag`, `model_task`,
`model_task_source`, `library_name`, `license`, `license_name`, `license_link`,
`base_model`, `base_model_relation`, `tags`, `parameters_total`, `dtype`,
`quant`, `architectures`, `model_type`, `context_length`, and `gated`.

`model_task` equals `pipeline_tag`; `model_task_source` is `model_card` or
`inferred`. `parameters_total` comes from `safetensors.total` or `gguf.total`;
`dtype` copies the safetensors dtype breakdown; `quant` copies the GGUF file
type or Hub quantization metadata. The publisher never resolves model metadata
from a moving branch head.

## Cloud endpoint records

A run may declare `model_kind: "cloud-endpoint"` with `model_provider` and no
`model_revision`. The publisher skips the Hub model lookup for that kind and
builds the model fields from the declared identity: `model_id` is
`<model_provider>/<model>`, and every Hub-derived field is null with a
`not_applicable` reason in `dimension_null_reasons`. The Parquet shard path
slugs the provider and model, and rows carry `model_kind` and `model_provider`
columns, so cloud-endpoint rows stay separable from local model rows and are
never summed with them. The schema only allows `model_provider` together with
`model_kind: "cloud-endpoint"`, which forbids `model_revision`. Local models
keep the strict Hub revision and lookup checks unchanged.

## Evaluation metadata

An evaluation row records `dataset_id`, `dataset_task_id`,
`dataset_revision`, `value`, `metric`, `date`, `source_url`, `notes`,
`evaluation_framework`, `config`, and `split`. Before publishing an evaluation
row, the publisher checks the dataset revision, its `benchmark` tag, its
`eval.yaml`, the registered framework, and the matching task id/config/split.

The Hub `.eval_results/*.yaml` file keeps Hugging Face's exact shape:
`dataset.id`, `dataset.task_id`, optional `dataset.revision`, `value`, `date`,
optional `source.url`, and optional `notes`. That format has no metric key, so
the metric remains in the Parquet row and is included in the sidecar notes.
Performance-only rows mark all dataset and evaluation fields
`not_applicable` and do not emit evaluation sidecars.

## Run metadata and nulls

Every row also contains `engine`, `engine_version`, `profile`, `hardware`,
`power_limit_w`, `concurrency`, `ctx_per_slot`, `kv_cache_dtype`, `thinking`,
`temperature`, `max_tokens`, `prompt_chars`, `prompt_tokens`,
`system_prompt_chars`, `system_prompt_tokens`, `runner`, `harness`,
`router_key_alias`, `run_id`, `start_utc`, and `end_utc`.

These exact field names are top-level Parquet columns. Composite values such
as `dtype`, array-valued `base_model`, `hardware`, and
`dimension_null_reasons` use JSON encoding in their own columns; `tags` and
`architectures` are Parquet lists. The separate system-engine object uses
`system_engine` so the requested `engine` column keeps its run-variable value.
The public projection removes host and network identifiers.

Every critical field is required. A null is accepted only with an entry in
`dimension_null_reasons`; the only codes are `not_applicable` and
`na_backfill`. `na_backfill` is rejected for rows after
`2026-10-07T04:15:00Z`. Tags, source URL, and notes are optional. If
`base_model_relation` is set, `base_model` must be provided or carry an
explicit null reason. If `license` is `other`, `license_name` and
`license_link` are required; a null must carry an allowed reason like every
other required field.

## Validation and migration

The same JSON Schema drives the publisher, `mlx-bench-validate-results`, and
the exported pre-commit hook. The repository's `Merge Gate` includes publisher
validation, and the reusable workflow is available from `JacobPEvans/.github` for
consumer repositories to add as a required status check.

`mlx-bench-publish-backfill` is the one-time migration. It reads every
historical Parquet shard and defaults to dry-run. It fills Hub fields only when
the row contains a recorded model SHA, preserves derivable run values, removes
host identifiers, and marks facts that cannot be derived as null with
`na_backfill`. It emits sidecars only for complete tasks registered by the
Hub. The command reports planned shard and sidecar counts without writing.

```sh
mlx-bench-publish-backfill
```

The lead reviews the full dry-run before authorizing the one-time dataset
write. `--apply` performs that write and must not be used during routine
validation.
