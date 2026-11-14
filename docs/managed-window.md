# Managed window (solo serving, isolated class)

When the model isn't in the `llama-swap` config, or you need the whole
machine's memory for one large model, take a **managed window** on the Studio:
solo `vllm-mlx serve` while production serving is paused. This takes
production serving (the Hermes brain) offline, so:

> **Notify the user before opening a managed window, and restore after.**
> Production Hermes is down for the duration.

With session approval for benchmark priority, log approval, window start/end,
services paused/restored, and readiness in the task. Approval never waives
paired, context, concurrency, or environment rules
([RUNBOOK Step 3](RUNBOOK.md#step-3--serve-the-model)).

```sh
# 1. Bootout serving AND everything that could relaunch it mid-window
#    (the 30s mlx-night.watcher agent; KeepAlive resurrects a plain `kill`).
for a in vllm-mlx.server mlx-night.watcher mlx-night.rank mlx-night.prefetch; do
  launchctl bootout "gui/501/dev.$a" 2>/dev/null || true
done
# Rotation flips (00:00Z/12:00Z) need no pause: their curls fail harmlessly.
# To freeze across flips, touch the per-router rotation-paused sentinel
# (apps docs/BRAIN_ROTATION.md). Permanent policy = ai_rotation_enabled.

# 2. Serve the target model solo (parser flags from the
#    [parser map](benchmark-traps.md#parser-map); size it per RUNBOOK Step 2)
vllm-mlx serve <model-id> \
  --port 11434 \
  --tool-call-parser <parser> \
  --reasoning-parser <parser> \
  --gpu-memory-utilization 0.80 \
  --timeout 3600

# 3. ... run your suites against http://127.0.0.1:11434/v1 ...

# 4. Restore (server first; the warmup agent re-faults the residents)
for a in vllm-mlx.server vllm-mlx.warmup mlx-night.watcher mlx-night.rank mlx-night.prefetch; do
  launchctl bootstrap gui/501 ~/Library/LaunchAgents/dev.$a.plist 2>/dev/null || true
done

# 5. Verify before closing the window: every resident "ready"
curl -s http://127.0.0.1:11434/running
```

Mind the [serving flags that bite](benchmark-traps.md#serving-flags-that-bite).
On the Studio, HF auth may be unset — a model that needs downloading reads
`HF_TOKEN` (cache on `/Volumes/HuggingFace`).
