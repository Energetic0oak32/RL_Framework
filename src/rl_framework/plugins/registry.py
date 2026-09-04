from __future__ import annotations

from typing import Dict

from .algorithm import AlgorithmPlugin
from .environment import EnvironmentPlugin
from .simulator import SimulatorPlugin


class PluginRegistry:
    """
    Registro central de plugins disponíveis no framework.

    Mantém separados:
    - plugins de algoritmo
    - plugins de ambiente
    - plugins de simulador
    """

    def __init__(self) -> None:
        self._algorithms: Dict[str, AlgorithmPlugin] = {}
        self._environments: Dict[str, EnvironmentPlugin] = {}
        self._simulators: Dict[str, SimulatorPlugin] = {}

    # ============================================================
    # Algorithm plugins
    # ============================================================

    def register_algorithm(
        self,
        plugin: AlgorithmPlugin,
    ) -> None:

        name = plugin.metadata.name

        if name in self._algorithms:
            raise ValueError(
                f"Algorithm plugin '{name}' is already registered."
            )

        self._algorithms[name] = plugin

    def get_algorithm(
        self,
        name: str,
    ) -> AlgorithmPlugin:

        try:
            return self._algorithms[name]

        except KeyError:
            raise KeyError(
                f"Algorithm plugin '{name}' was not found."
            ) from None

    def algorithms(
        self,
    ) -> list[AlgorithmPlugin]:

        return list(self._algorithms.values())

    # ============================================================
    # Environment plugins
    # ============================================================

    def register_environment(
        self,
        plugin: EnvironmentPlugin,
    ) -> None:

        name = plugin.metadata.name

        if name in self._environments:
            raise ValueError(
                f"Environment plugin '{name}' is already registered."
            )

        self._environments[name] = plugin

    def get_environment(
        self,
        name: str,
    ) -> EnvironmentPlugin:

        try:
            return self._environments[name]

        except KeyError:
            raise KeyError(
                f"Environment plugin '{name}' was not found."
            ) from None

    def environments(
        self,
    ) -> list[EnvironmentPlugin]:

        return list(self._environments.values())

    # ============================================================
    # Simulator plugins
    # ============================================================

    def register_simulator(
        self,
        plugin: SimulatorPlugin,
    ) -> None:

        name = plugin.metadata.name

        if name in self._simulators:
            raise ValueError(
                f"Simulator plugin '{name}' is already registered."
            )

        self._simulators[name] = plugin

    def get_simulator(
        self,
        name: str,
    ) -> SimulatorPlugin:

        try:
            return self._simulators[name]

        except KeyError:
            raise KeyError(
                f"Simulator plugin '{name}' was not found."
            ) from None

    def simulators(
        self,
    ) -> list[SimulatorPlugin]:

        return list(self._simulators.values())

    # ============================================================
    # General
    # ============================================================

    def clear(self) -> None:
        """
        Remove todos os plugins registrados.

        Útil principalmente para testes.
        """

        self._algorithms.clear()
        self._environments.clear()
        self._simulators.clear()