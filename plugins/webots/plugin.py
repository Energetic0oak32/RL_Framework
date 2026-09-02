from __future__ import annotations

from typing import Any

from rl_framework.plugins.base import (
    PluginCapabilities,
    PluginMetadata,
)
from rl_framework.plugins.environment import EnvironmentPlugin
from rl_framework.plugins.registry import PluginRegistry


class DummyWebotsPlugin(EnvironmentPlugin):

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Test Webots",
            version="0.1.0",
            description="Temporary Webots plugin used to test discovery.",
            author="RL Framework",
        )

    @property
    def capabilities(self) -> PluginCapabilities:
        return PluginCapabilities(
            parallel_instances=True,
            supported_action_spaces=[
                "Discrete",
                "Box",
            ],
        )

    def config_schema(self) -> dict[str, Any]:
        return {
            "world": {
                "type": "string",
                "default": "",
            }
        }

    def create(
        self,
        config: dict[str, Any],
        instances: int = 1,
    ) -> Any:

        return {
            "type": "dummy_webots_environment",
            "instances": instances,
            "config": config,
        }

    def close(
        self,
        environment: Any,
    ) -> None:

        print("Dummy Webots environment closed.")


def register(
    registry: PluginRegistry,
) -> None:

    registry.register_environment(
        DummyWebotsPlugin()
    )