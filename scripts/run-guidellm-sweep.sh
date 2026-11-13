#!/usr/bin/env bash
# Run the reasoning-on GuideLLM concurrency sweep for one served model.
#
# Thin wrapper around configs/guidellm/thinking-on-sweep.toml. It sequences
# one `guidellm run` invocation per (concurrency level, repeat) pair — each
# writes a single-benchmark benchmarks.json that `mlx-bench-publish --kind
# guidellm` converts. It does NOT reimplement load generation and does NOT
# orchestrate serving: point it at an endpoint that already serves $MODEL
# within its deployed concurrency limit.
#
# GuideLLM is an external tool (not a dependency of this repo):
#   pip install "guidellm[recommended]"   # CLI verified against v0.8.1 docs
#
# Usage:
#   scripts/run-guidellm-sweep.sh --base-url URL --model ID \
#     --api-key-env VAR_NAME --output-dir DIR [options]
#
#   --base-url URL          server ROOT, no /v1 path (GuideLLM appends the
#                           request_format path itself — unlike the repo's
#                           other tools)
#   --model ID              model id as advertised by the endpoint
#   --api-key-env NAME      NAME of the env var holding the API key; the key
#                           value is passed to guidellm but never echoed.
#                           Omit the flag for endpoints that need no key.
#   --output-dir DIR        one JSON report per (level, repeat) is written here
#   --levels 1,2,4,8        concurrency levels (default matches the config)
#   --duration-seconds N    fixed measured window per level (default 300)
#   --repeats N             consecutive runs per level (default 2; the
#                           verdict policy requires a pair, not one run)
#   --prompt-tokens N       synthetic prompt target (default 1000)
#   --max-tokens N          generation cap so reasoning + answer fit (default 8192)
#   --dry-run               print the commands; run nothing
#
# After the sweep, publish each report (dry-run first):
#   .venv/bin/mlx-bench-publish <output-dir>/guidellm_*_c<C>_r<R>.json \
#     --kind guidellm --suite throughput --model <model-id> \
#     --env-class <isolated|under-load> \
#     --published-metadata run-output/published-metadata.json --dry-run
set -euo pipefail

BASE_URL=""
MODEL=""
API_KEY_ENV=""
OUTPUT_DIR="run-output/guidellm-sweep"
LEVELS="1,2,4,8"
DURATION=300
REPEATS=2
PROMPT_TOKENS=1000
MAX_TOKENS=8192
DRY_RUN=0

die() {
  echo "run-guidellm-sweep: $*" >&2
  exit 1
}

usage() {
  sed -n '2,33p' "$0"
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --api-key-env) API_KEY_ENV="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --levels) LEVELS="$2"; shift 2 ;;
    --duration-seconds) DURATION="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --prompt-tokens) PROMPT_TOKENS="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
done

[ -n "$BASE_URL" ] || die "--base-url is required (server root, no /v1 path)"
[ -n "$MODEL" ] || die "--model is required"
case "$BASE_URL" in
  */v1|*/v1/) die "--base-url must be the server ROOT without /v1 — GuideLLM appends the request path" ;;
esac
if [ "$DRY_RUN" -eq 0 ] && ! command -v guidellm >/dev/null 2>&1; then
  die "guidellm not found on PATH (pip install 'guidellm[recommended]')"
fi

# The key VALUE travels only inside the child process argv (GuideLLM's
# documented api_key backend arg); this script never prints it. An unset or
# empty var means no key — localhost serving endpoints usually need none.
API_KEY_VALUE=""
if [ -n "$API_KEY_ENV" ]; then
  API_KEY_VALUE="${!API_KEY_ENV:-}"
  [ -n "$API_KEY_VALUE" ] || echo "run-guidellm-sweep: \$$API_KEY_ENV is unset — running without an API key" >&2
fi

SLUG="$(echo "$MODEL" | tr '/.' '__')"
mkdir -p "$OUTPUT_DIR"

backend_args="kind=openai_http,target=$BASE_URL,model=$MODEL,request_format=/v1/chat/completions"
backend_args="$backend_args,extras.body.max_tokens=$MAX_TOKENS,extras.body.ignore_eos=false"
if [ -n "$API_KEY_VALUE" ]; then
  backend_args="$backend_args,api_key=$API_KEY_VALUE"
fi

IFS=',' read -r -a LEVEL_LIST <<< "$LEVELS"
for LEVEL in "${LEVEL_LIST[@]}"; do
  case "$LEVEL" in
    ''|*[!0-9]*) die "invalid concurrency level: $LEVEL" ;;
  esac
  for REP in $(seq 1 "$REPEATS"); do
    OUT="$OUTPUT_DIR/guidellm_${SLUG}_c${LEVEL}_r${REP}.json"
    echo "run-guidellm-sweep: level=$LEVEL repeat=$REP -> $OUT"
    if [ "$DRY_RUN" -eq 1 ]; then
      echo "  would run: guidellm run --backend <openai_http: target=$BASE_URL model=$MODEL chat completions, cap $MAX_TOKENS tokens, eos respected$( [ -n "$API_KEY_VALUE" ] && echo ', with api key')> \\"
      echo "    --profile kind=concurrent,streams=$LEVEL,warmup=0.1,cooldown=0.05 \\"
      echo "    --constraint kind=max_duration,seconds=$DURATION \\"
      echo "    --data kind=synthetic_text,prompt_tokens=$PROMPT_TOKENS \\"
      echo "    --seed kind=static,value=$REP --output kind=json,path=$OUT"
      continue
    fi
    guidellm run \
      --backend "$backend_args" \
      --profile "kind=concurrent,streams=$LEVEL,warmup=0.1,cooldown=0.05" \
      --constraint "kind=max_duration,seconds=$DURATION" \
      --data "kind=synthetic_text,prompt_tokens=$PROMPT_TOKENS" \
      --seed "kind=static,value=$REP" \
      --output "kind=json,path=$OUT"
  done
done

echo "run-guidellm-sweep: reports in $OUTPUT_DIR"
echo "run-guidellm-sweep: next: mlx-bench-publish <report>.json --kind guidellm --suite throughput --dry-run"
