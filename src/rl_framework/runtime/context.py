from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rl_framework.core.experiments import Experiment


@dataclass(slots=True)
class TrainingContext:
    """
    Informações do framework disponíveis durante
    a execução de um treinamento.

    No futuro poderá conter:
    - callbacks
    - metric sinks
    - checkpoint manager
    - stop signals
    - logging
    - run id
    """

    experiment: Experiment
    output_path: Path