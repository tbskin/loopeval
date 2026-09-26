"""LoopEval public API."""

from .api import LoopEval
from .bootstrap import propose_initial_checks
from .config import LoopEvalConfig, load_config
from .models import (
    BootstrapCandidate,
    BootstrapProposal,
    CandidateCheck,
    CheckResult,
    CheckSpec,
    EvalSample,
    EvaluationReport,
    SampleResult,
)
from .providers import DecisionProvider, GenerativeProvider

__all__ = [
    "BootstrapCandidate",
    "BootstrapProposal",
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
    "propose_initial_checks",
]

__version__ = "0.1.0"
