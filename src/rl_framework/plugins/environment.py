from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from .base import BasePlugin


@dataclass(frozen=True, slots=True)
class EnvironmentSpec:
    """
    Descreve as características do ambiente de RL
    relevantes para compatibilidade com algoritmos.
    """

    action_space: str
    observation_space: str


class EnvironmentPlugin(BasePlugin):
    """
    Interface base para ambientes de reinforcement learning.

    Um EnvironmentPlugin representa a tarefa de RL propriamente dita:
    action space, observation space, reward, reset, step etc.

    O ambiente pode ou não depender de um SimulatorPlugin.
    """

    @property
    def required_simulator(self) -> str | None:
        """
        Nome do simulador necessário para executar este ambiente.

        Retorna None quando o ambiente não depende
        de simulador externo.
        """

        return None

    @abstractmethod
    def environment_spec(
        self,
        config: dict[str, Any],
    ) -> EnvironmentSpec:
        """
        Retorna as características do ambiente após
        considerar sua configuração.

        Isso permite que um mesmo plugin altere seus
        spaces dependendo da configuração utilizada.
        """

        pass

    @abstractmethod
    def create(
        self,
        config: dict[str, Any],
        simulator: Any | None = None,
        instances: int = 1,
    ) -> Any:
        """
        Cria o ambiente de reinforcement learning.
        """

        pass

    @abstractmethod
    def close(
        self,
        environment: Any,
    ) -> None:
        """
        Fecha o ambiente e libera seus recursos.
        """

        pass