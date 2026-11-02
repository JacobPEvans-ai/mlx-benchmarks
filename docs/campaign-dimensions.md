# Campaign dimensions

`campaign_dimensions` is optional so legacy v1 envelopes remain valid. Its
optional groups contain the typed fields defined in `schema.json`; every leaf
is optional and accepts explicit `null` when a measurement is unavailable or
inapplicable. The publisher flattens each leaf into one stable
`campaign_<group>_<field>` Parquet column. Environment fan duty, patches, knob
maps, sampling settings, build architecture, build-flag objects, and
throttle-reason fields use a `_json` column suffix. Older canonical shards are
normalized with nulls for all new columns, so every file has the same schema.

`serving.stack` accepts `direct` for runs sent directly to an engine endpoint.

## Field groups

- **hardware:** machine, interconnect, and world size; accelerator model,
  memory, and bandwidth; host CPU,
  RAM, and speed; PCIe generation/width; power cap; UPS/circuit;
  chassis/container; optional published inventory device ID.
- **environment:** ambient temperature in Fahrenheit; fan duty for GPU,
  chassis, and Mac controllers.
- **software:** OS; kernel; driver; accelerator runtime; engine/version/commit;
  backend; GPU architectures; build flags; flash attention; CUDA Toolkit version.
- **model:** family/ID/HF repo/revision; total/active parameters; architecture;
  total/active experts; quantization; bits per weight; file size; license;
  native context; single-file SHA-256.
- **run:** allocated context; KV dtype; prompt/depth/output tokens; concurrency;
  batch/ubatch; parallel slots; prefix cache; speculative/MTP; draft model;
  temperature; thinking; chat template; seed; repeats; warm/cold state; cell
  sample-slice index; start/end; applied patches; `knob_case` and structured `knobs`; sampling mode;
  requested/measured output tokens; speculative-token count; KV-pool tokens;
  max running requests; PLE offload target; model load time.
- **speed:** TTFT p50/p90/p99; prefill/decode/aggregate/total throughput;
  TPOT; ITL p50/p99; MTP acceptance and accepted length; KV occupancy;
  prefix-cache hit rate; queue-time p50/p99; goodput; `c_at_p10_35tps`;
  request success rate.
- **resource:** accelerator/host memory peak; CPU offload; average/max GPU
  utilization; PLE bytes; accelerator/system/SoC/component power; energy per
  token; tokens per watt; temperature; GPU/CPU clocks; throttling; fan.
- **quality:** benchmark/version/subset/sample count; score; standard error and confidence interval; judge model; contamination note.
- **provenance:** UTC timestamp; run ID; config name/SHA; operator/agent; raw output path; schema version.
- **cost:** kWh and local electricity cost per million output tokens; median rental and API cost per million.

`campaign_dimensions.run.sample_slice_index` identifies the sample-slice cell
within its recipe and is flattened to the stable
`campaign_run_sample_slice_index` Parquet column.

## Optional power enrichment

Set `MLX_BENCH_METRICS_URL` to a Prometheus or VictoriaMetrics base URL. Paths
ending in `/api/v1` or `/api/v1/query_range` are also accepted. The publisher
uses cell `run.started_at` and `run.ended_at`; if either is absent, it uses the
envelope timestamp with `run.phase_duration_seconds`, or the longest result
duration, to determine the query window.

The publisher queries `/api/v1/query_range` at one-second steps with
`sum(DCGM_FI_DEV_POWER_USAGE)` for NVIDIA systems or
`macmon_sys_power_watts` for macOS. The range mean and maximum populate the
existing `resource.power_avg_w` and `resource.power_max_w` fields. Trapezoidal
integration divided by measured output tokens derives
`energy_j_per_output_token`; output tokens per second divided by average watts
derives `tokens_per_second_per_watt`. If the URL is unset or output-token count
is unavailable, affected nullable fields receive `dimension_null_reasons`
codes. A configured query that fails aborts publish.

The Mac metric represents whole-system power. Record the measurement scope in
result tags when comparing it with accelerator-only telemetry.

References: [Prometheus query API](https://prometheus.io/docs/prometheus/latest/querying/api/),
[VictoriaMetrics range query](https://docs.victoriametrics.com/victoriametrics/keyconcepts/#range-query),
[macmon Prometheus exporter](https://github.com/vladkens/macmon/blob/main/src_app/serve.rs).
