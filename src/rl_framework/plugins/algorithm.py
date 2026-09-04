from __future__ import annotations

from abc import abstractmethod
from typing import Any

from .base import BasePlugin


class AlgorithmPlugin(BasePlugin):
    """
    Interface para plugins de algoritmos de treinamento.

    Exemplos:
        Stable-Baselines3 PPO
        Stable-Baselines3 DQN
        Stable-Baselines3 SAC
        CleanRL PPO
        algoritmos customizados
    """

    @abstractmethod
    def create(
        self,
        environment: Any,
        config: dict[str, Any],
    ) -> Any:
        """
        Cria o modelo/algoritmo usando o ambiente fornecido.
        """
        pass

    @abstractmethod
    def train(
        self,
        model: Any,
        config: dict[str, Any],
        context: Any,
    ) -> None:
        """
        Executa o treinamento.
        """
        pass

    @abstractmethod
    def save(
        self,
        model: Any,
        path: str,
    ) -> None:
        """
        Salva o modelo treinado.
        """
        pass
    
    @abstractmethod
    def load(
        self,
        path: str,
        environment: Any | None = None,
    ) -> Any:
        """
        Carrega um modelo previamente salvo.
        """

        pass


    @abstractmethod
    def predict(
        self,
        model: Any,
        observation: Any,
        deterministic: bool = True,
    ) -> Any:
        """
        Produz uma ação usando um modelo treinado.
        """

        pass