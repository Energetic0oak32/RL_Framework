from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Experiment:
    """
    Descrição de um experimento de reinforcement learning.

    Esta classe não executa o treinamento diretamente.
    Ela apenas descreve quais plugins e configurações
    devem ser utilizados.
    """

    algorithm: str
    environment: str

    instances: int = 1

    algorithm_config: dict[str, Any] = field(
        default_factory=dict
    )

    environment_config: dict[str, Any] = field(
        default_factory=dict
    )

    simulator_config: dict[str, Any] = field(
        default_factory=dict
    )

    output_path: Path = Path("runs/model")

    def __post_init__(self) -> None:
        if self.instances < 1:
            raise ValueError(
                "Experiment instances must be at least 1."
            )

        self.output_path = Path(self.output_path)