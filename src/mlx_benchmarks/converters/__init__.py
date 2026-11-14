"""Converters transform raw tool output into envelope v1."""

from mlx_benchmarks.converters.agentic import AgenticConverter
from mlx_benchmarks.converters.agentic_partial import AgenticPartialConverter
from mlx_benchmarks.converters.base import Converter, ConverterContext
from mlx_benchmarks.converters.bench_serve import BenchServeConverter
from mlx_benchmarks.converters.coding_replay import CodingReplayConverter
from mlx_benchmarks.converters.dense_layout import DenseLayoutConverter
from mlx_benchmarks.converters.evalscope import EvalScopeConverter
from mlx_benchmarks.converters.factual import FactualConverter
from mlx_benchmarks.converters.fio import FioConverter
from mlx_benchmarks.converters.gpu_burn import GpuBurnConverter
from mlx_benchmarks.converters.guidellm import GuidellmConverter
from mlx_benchmarks.converters.inspect_ai import InspectAIConverter
from mlx_benchmarks.converters.llamacpp_server import LlamaCppServerConverter
from mlx_benchmarks.converters.lm_eval import LmEvalConverter
from mlx_benchmarks.converters.mbw import MbwConverter
from mlx_benchmarks.converters.nvbandwidth import NvbandwidthConverter
from mlx_benchmarks.converters.nvidia_offload import KvOffloadConverter, MoeOffloadConverter
from mlx_benchmarks.converters.promptfoo import PromptfooConverter
from mlx_benchmarks.converters.promptstack import PromptstackConverter
from mlx_benchmarks.converters.tau2 import Tau2Converter
from mlx_benchmarks.converters.throughput_probe import ThroughputProbeConverter
from mlx_benchmarks.converters.vllm import VllmConverter

__all__ = [
    "AgenticConverter",
    "AgenticPartialConverter",
    "BenchServeConverter",
    "CodingReplayConverter",
    "Converter",
    "ConverterContext",
    "DenseLayoutConverter",
    "EvalScopeConverter",
    "FactualConverter",
    "FioConverter",
    "GpuBurnConverter",
    "GuidellmConverter",
    "InspectAIConverter",
    "KvOffloadConverter",
    "LlamaCppServerConverter",
    "LmEvalConverter",
    "MbwConverter",
    "MoeOffloadConverter",
    "NvbandwidthConverter",
    "PromptfooConverter",
    "PromptstackConverter",
    "Tau2Converter",
    "ThroughputProbeConverter",
    "VllmConverter",
    "get_converter",
]


def get_converter(kind: str) -> Converter:
    """Return the converter registered for ``kind``.

    Raises :class:`ValueError` for unknown kinds so callers get a clear signal
    instead of silently defaulting.
    """
    registry: dict[str, type[Converter]] = {
        "agentic": AgenticConverter,
        "agentic-partial": AgenticPartialConverter,
        "bench-serve": BenchServeConverter,
        "coding-replay": CodingReplayConverter,
        "dense-layout": DenseLayoutConverter,
        "evalscope": EvalScopeConverter,
        "factual": FactualConverter,
        "fio": FioConverter,
        "guidellm": GuidellmConverter,
        "gpu-burn": GpuBurnConverter,
        "inspect-ai": InspectAIConverter,
        "llamacpp-server": LlamaCppServerConverter,
        "lm-eval": LmEvalConverter,
        "kv-offload": KvOffloadConverter,
        "mbw": MbwConverter,
        "moe-offload": MoeOffloadConverter,
        "nvbandwidth": NvbandwidthConverter,
        "promptstack": PromptstackConverter,
        "promptfoo": PromptfooConverter,
        "tau2": Tau2Converter,
        "throughput-probe": ThroughputProbeConverter,
        "vllm": VllmConverter,
    }
    try:
        cls = registry[kind]
    except KeyError as exc:
        known = ", ".join(sorted(registry)) or "(none)"
        raise ValueError(f"Unknown converter kind {kind!r}; known kinds: {known}") from exc
    return cls()
