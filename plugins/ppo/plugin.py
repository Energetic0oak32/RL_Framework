from __future__ import annotations

from typing import Any

from rl_framework.plugins.algorithm import AlgorithmPlugin
from rl_framework.plugins.base import (
    PluginCapabilities,
    PluginMetadata,
)
from rl_framework.plugins.registry import PluginRegistry


class DummyPPOPlugin(AlgorithmPlugin):

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Test PPO",
            version="0.1.0",
            description="Temporary PPO plugin used to test discovery.",
            author="RL Framework",
        )

    @property
    def capabilities(self) -> PluginCapabilities:
        return PluginCapabilities(
            parallel_instances=True,
            checkpointing=True,
            supported_action_spaces=[
                "Discrete",
                "Box",
            ],
        )

    def config_schema(self) -> dict[str, Any]:
        return {
            "learning_rate": {
                "type": "float",
                "default": 0.0003,
            },
            "n_steps": {
                "type": "int",
                "default": 2048,
            },
        }

    def create(
        self,
        environment: Any,
        config: dict[str, Any],
    ) -> Any:

        return {
            "type": "dummy_ppo_model",
            "environment": environment,
            "config": config,
        }

    def train(
        self,
        model: Any,
        config: dict[str, Any],
        context: Any,
    ) -> None:

        print("Dummy PPO training started.")

    def save(
        self,
        model: Any,
        path: str,
    ) -> None:

        print(f"Dummy PPO model saved to: {path}")


def register(
    registry: PluginRegistry,
) -> None:

    registry.register_algorithm(
        DummyPPOPlugin()
    )