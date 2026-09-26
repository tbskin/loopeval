"""LoopEval public API."""

from .api import LoopEval
from .config import LoopEvalConfig, load_config
from .models import (
    CandidateCheck,
    CheckResult,
    CheckSpec,
    EvalSample,
    EvaluationReport,
    SampleResult,
)
from .providers import DecisionProvider, GenerativeProvider

__all__ = [
    "CandidateCheck",
    "CheckResult",
    "CheckSpec",
    "DecisionProvider",
    "EvalSample",
    "EvaluationReport",
    "GenerativeProvider",
    "LoopEval",
    "LoopEvalConfig",
    "SampleResult",
    "load_config",
]

__version__ = "0.1.0"
