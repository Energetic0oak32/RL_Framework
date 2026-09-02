from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PluginMetadata:
    name: str
    version: str
    description: str = ""
    author: str = ""


@dataclass
class PluginCapabilities:
    parallel_instances: bool = False
    checkpointing: bool = False

    supported_action_spaces: list[str] = field(default_factory=list)
    supported_observation_spaces: list[str] = field(default_factory=list)


class BasePlugin(ABC):
    """
    Base comum para todos os plugins do framework.
    """

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        pass

    @property
    @abstractmethod
    def capabilities(self) -> PluginCapabilities:
        pass

    @abstractmethod
    def config_schema(self) -> dict[str, Any]:
        """
        Retorna os parâmetros configuráveis do plugin.

        A GUI usará esse schema para gerar os campos
        automaticamente.
        """
        pass