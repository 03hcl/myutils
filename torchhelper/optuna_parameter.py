from enum import auto, Enum
from typing import Any, Dict, Generic, TypeVar

from optuna.trial import Trial

T = TypeVar("T")


class OptunaSuggestion(Enum):
    Categorical = auto()
    DiscreteUniform = auto()
    Int = auto()
    LogUniform = auto()
    Uniform = auto()


class OptunaParameter(Generic[T]):

    def __init__(self, suggestion_type: OptunaSuggestion, name: str, low: T, high: T, **kwargs: Any):
        self.suggestion_type: OptunaSuggestion = suggestion_type
        self.name: str = name
        self.low: T = low
        self.high: T = high
        self.kwargs: Dict[str, Any] = kwargs

    def get_optuna_parameter(self, trial: Trial):
        if self.suggestion_type == OptunaSuggestion.DiscreteUniform:
            return trial.suggest_discrete_uniform(self.name, self.low, self.high, self.kwargs["q"])
        elif self.suggestion_type == OptunaSuggestion.Int:
            return trial.suggest_int(self.name, self.low, self.high)
        elif self.suggestion_type == OptunaSuggestion.LogUniform:
            return trial.suggest_loguniform(self.name, self.low, self.high)
        elif self.suggestion_type == OptunaSuggestion.Uniform:
            return trial.suggest_uniform(self.name, self.low, self.high)
        else:
            raise NotImplementedError
