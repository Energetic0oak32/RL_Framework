from __future__ import annotations

from abc import abstractmethod
from typing import Any

from .base import BasePlugin


class EnvironmentPlugin(BasePlugin):
    """
    Interface para plugins responsáveis por criar ambientes.

    O ambiente retornado deverá, inicialmente, seguir a
    interface do Gymnasium.
    """

    @abstractmethod
    def create(
        self,
        config: dict[str, Any],
        instances: int = 1,
    ) -> Any:
        """
        Cria uma ou mais instâncias do ambiente.
        """
        pass

    @abstractmethod
    def close(
        self,
        environment: Any,
    ) -> None:
        """
        Finaliza o ambiente e libera seus recursos.
        """
        pass