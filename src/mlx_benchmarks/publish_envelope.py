"""``mlx-bench-publish-envelope`` command-line entry point.

Publishes committed envelope/metadata pairs, one ``--pair`` per run::

    mlx-bench-publish-envelope \\
        --pair published/pending/run-1/envelope.json published/pending/run-1/metadata.json \\
        --dry-run

Every pair is attempted even if an earlier one fails; the exit status is 1
when any pair fails and the log names the files involved.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, cast

from mlx_benchmarks.envelope import Envelope, validate_envelope
from mlx_benchmarks.publish import DEFAULT_REPO_ID, PublishError, publish

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-bench-publish-envelope",
        description="Publish committed envelope/metadata pairs to the HF dataset.",
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        required=True,
        metavar=("ENVELOPE_JSON", "METADATA_JSON"),
        help="envelope JSON and its published-metadata JSON; repeat for each run",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate + plan only; do not upload")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF dataset repo to publish to")
    return parser


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _publish_pair(envelope_path: Path, metadata_path: Path, *, dry_run: bool, repo_id: str) -> str:
    envelope = cast(Envelope, _load_object(envelope_path))
    metadata = _load_object(metadata_path)
    validate_envelope(envelope)
    return publish(envelope, published_metadata=metadata, dry_run=dry_run, repo_id=repo_id)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Match mlx-bench-publish: UTC timestamps ending in Z.
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    failures = 0
    for envelope_arg, metadata_arg in args.pair:
        envelope_path, metadata_path = Path(envelope_arg), Path(metadata_arg)
        try:
            path = _publish_pair(envelope_path, metadata_path, dry_run=args.dry_run, repo_id=args.repo_id)
        except (OSError, ValueError, PublishError) as exc:
            log.error("%s with %s failed: %s", envelope_path, metadata_path, exc)
            failures += 1
            continue
        log.info("%s -> %s", envelope_path, path)
    return 1 if failures else 0
