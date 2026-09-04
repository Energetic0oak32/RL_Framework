from __future__ import annotations

from typing import Any

from rl_framework.plugins.base import (
    PluginCapabilities,
    PluginMetadata,
)
from rl_framework.plugins.environment import (
    EnvironmentPlugin,
    EnvironmentSpec,
)
from rl_framework.plugins.registry import PluginRegistry


class DummyEnvironmentPlugin(EnvironmentPlugin):

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Test Environment",
            version="0.1.0",
            description=(
                "Temporary RL environment used to test "
                "environment/simulator composition."
            ),
            author="RL Framework",
        )

    @property
    def capabilities(self) -> PluginCapabilities:
        return PluginCapabilities(
            parallel_instances=True,
        )

    @property
    def required_simulator(self) -> str | None:
        return "Test Webots"

    def config_schema(self) -> dict[str, Any]:
        return {
            "example_parameter": {
                "type": "float",
                "default": 1.0,
            }
        }

    def environment_spec(
        self,
        config: dict[str, Any],
    ) -> EnvironmentSpec:

        return EnvironmentSpec(
            action_space="Box",
            observation_space="Box",
        )

    def create(
        self,
        config: dict[str, Any],
        simulator: Any | None = None,
        instances: int = 1,
    ) -> Any:

        if simulator is None:
            raise RuntimeError(
                "Test Environment requires a simulator."
            )

        return {
            "type": "dummy_rl_environment",
            "simulator": simulator,
            "instances": instances,
            "config": config,
        }

    def close(
        self,
        environment: Any,
    ) -> None:

        print("Dummy RL environment closed.")


def register(
    registry: PluginRegistry,
) -> None:

    registry.register_environment(
        DummyEnvironmentPlugin()
    )