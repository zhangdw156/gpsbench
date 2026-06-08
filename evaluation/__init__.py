"""
GPSBench Evaluation Framework.

This trimmed evaluation harness only requires the LLM client at runtime.  Older
copies of the project also shipped task_evaluators.py; keep those imports
optional so package discovery/imports do not fail in the minimal fork.
"""

from .llm_client import LLMClient, LLMResponse, get_available_models

try:
    from .task_evaluators import (
        TaskEvaluator,
        FormatRecognitionEvaluator,
        FormatConversionEvaluator,
        ValidationEvaluator,
        CoordinateSystemEvaluator,
        PrecisionEvaluator,
        get_evaluator,
    )
except ModuleNotFoundError:
    TaskEvaluator = None
    FormatRecognitionEvaluator = None
    FormatConversionEvaluator = None
    ValidationEvaluator = None
    CoordinateSystemEvaluator = None
    PrecisionEvaluator = None

    def get_evaluator(*args, **kwargs):
        raise ModuleNotFoundError(
            "task_evaluators.py is not included in this trimmed GPSBench harness"
        )

__all__ = [
    "LLMClient",
    "LLMResponse",
    "get_available_models",
    "TaskEvaluator",
    "FormatRecognitionEvaluator",
    "FormatConversionEvaluator",
    "ValidationEvaluator",
    "CoordinateSystemEvaluator",
    "PrecisionEvaluator",
    "get_evaluator",
]

__version__ = "1.0.0"
