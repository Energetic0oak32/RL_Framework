from __future__ import annotations

from typing import Any

import gymnasium as gym

from rl_framework.plugins.base import (
    PluginCapabilities,
    PluginMetadata,
)
from rl_framework.plugins.environment import (
    EnvironmentPlugin,
    EnvironmentSpec,
)
from rl_framework.plugins.registry import PluginRegistry


class CartPoleEnvironmentPlugin(EnvironmentPlugin):

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="CartPole",
            version="1.0.0",
            description=(
                "Gymnasium CartPole-v1 reinforcement "
                "learning environment."
            ),
            author="RL Framework",
        )

    @property
    def capabilities(self) -> PluginCapabilities:
        return PluginCapabilities(
            parallel_instances=False,
        )

    @property
    def required_simulator(self) -> str | None:
        return None

    def config_schema(self) -> dict[str, Any]:
        return {}

    def environment_spec(
        self,
        config: dict[str, Any],
    ) -> EnvironmentSpec:

        return EnvironmentSpec(
            action_space="Discrete",
            observation_space="Box",
        )

    def create(
        self,
        config: dict[str, Any],
        simulator: Any | None = None,
        instances: int = 1,
    ) -> Any:

        if instances != 1:
            raise RuntimeError(
                "CartPole plugin currently supports "
                "only one environment instance."
            )

        return gym.make(
            "CartPole-v1"
        )

    def close(
        self,
        environment: Any,
    ) -> None:

        environment.close()


def register(
    registry: PluginRegistry,
) -> None:

    registry.register_environment(
        CartPoleEnvironmentPlugin()
    )