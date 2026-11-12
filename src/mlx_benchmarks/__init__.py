"""mlx-benchmarks — envelope v1 toolkit and HF publisher."""

from mlx_benchmarks.envelope import (
    Envelope,
    EnvelopeValidationError,
    Result,
    System,
    validate_envelope,
)

__all__ = [
    "Envelope",
    "EnvelopeValidationError",
    "Result",
    "System",
    "validate_envelope",
]

__version__ = "0.69.1"  # x-release-please-version
