from __future__ import annotations

from typing import Any

from rl_framework.plugins.base import (
    PluginCapabilities,
    PluginMetadata,
)
from rl_framework.plugins.registry import PluginRegistry
from rl_framework.plugins.simulator import SimulatorPlugin


class DummyWebotsPlugin(SimulatorPlugin):

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Test Webots",
            version="0.1.0",
            description=(
                "Temporary Webots simulator plugin "
                "used to test discovery."
            ),
            author="RL Framework",
        )

    @property
    def capabilities(self) -> PluginCapabilities:
        return PluginCapabilities(
            parallel_instances=True,
        )

    def config_schema(self) -> dict[str, Any]:
        return {
            "world": {
                "type": "string",
                "default": "",
            }
        }

    def launch(
        self,
        config: dict[str, Any],
        instances: int = 1,
    ) -> Any:

        return {
            "type": "dummy_webots_simulator",
            "instances": instances,
            "config": config,
        }

    def close(
        self,
        simulator: Any,
    ) -> None:

        print("Dummy Webots simulator closed.")


def register(
    registry: PluginRegistry,
) -> None:

    registry.register_simulator(
        DummyWebotsPlugin()
    )