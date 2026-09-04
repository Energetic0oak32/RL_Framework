from __future__ import annotations

from abc import abstractmethod
from typing import Any

from .base import BasePlugin


class SimulatorPlugin(BasePlugin):
    """
    Interface para plugins responsáveis por controlar simuladores.

    Um simulador não representa, por si só, um ambiente de
    reinforcement learning. Ele fornece a infraestrutura onde
    um EnvironmentPlugin pode executar.
    """

    @abstractmethod
    def launch(
        self,
        config: dict[str, Any],
        instances: int = 1,
    ) -> Any:
        """
        Inicia uma ou mais instâncias do simulador.
        """

        pass

    @abstractmethod
    def close(
        self,
        simulator: Any,
    ) -> None:
        """
        Finaliza as instâncias do simulador e libera seus recursos.
        """

        pass