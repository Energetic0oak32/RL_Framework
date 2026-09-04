from __future__ import annotations

from typing import Any

from stable_baselines3 import PPO

from rl_framework.plugins.algorithm import AlgorithmPlugin
from rl_framework.plugins.base import (
    PluginCapabilities,
    PluginMetadata,
)
from rl_framework.plugins.registry import PluginRegistry


class StableBaselines3PPOPlugin(AlgorithmPlugin):

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="PPO",
            version="1.0.0",
            description=(
                "Proximal Policy Optimization implementation "
                "provided by Stable-Baselines3."
            ),
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
                "MultiDiscrete",
                "MultiBinary",
            ],
            supported_observation_spaces=[
                "Box",
            ],
        )

    def config_schema(self) -> dict[str, Any]:
        return {

            "policy": {
                "type": "str",
                "default": "MlpPolicy",
                "choices": [
                    "MlpPolicy",
                ],
            },

            "learning_rate": {
                "type": "float",
                "default": 0.0003,
                "min": 0.0,
            },

            "n_steps": {
                "type": "int",
                "default": 2048,
                "min": 2,
            },

            "batch_size": {
                "type": "int",
                "default": 64,
                "min": 2,
            },

            "n_epochs": {
                "type": "int",
                "default": 10,
                "min": 1,
            },

            "gamma": {
                "type": "float",
                "default": 0.99,
                "min": 0.0,
                "max": 1.0,
            },

            "gae_lambda": {
                "type": "float",
                "default": 0.95,
                "min": 0.0,
                "max": 1.0,
            },

            "clip_range": {
                "type": "float",
                "default": 0.2,
                "min": 0.0,
            },

            "ent_coef": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
            },

            "vf_coef": {
                "type": "float",
                "default": 0.5,
                "min": 0.0,
            },

            "max_grad_norm": {
                "type": "float",
                "default": 0.5,
                "min": 0.0,
            },

            "total_timesteps": {
                "type": "int",
                "default": 10_000,
                "min": 1,
            },

            "verbose": {
                "type": "int",
                "default": 1,
                "min": 0,
                "max": 2,
            },

            "device": {
                "type": "str",
                "default": "auto",
            },

            "reset_num_timesteps": {
                "type": "bool",
                "default": True,
            },
        }

    def create(
        self,
        environment: Any,
        config: dict[str, Any],
    ) -> PPO:

        return PPO(
            policy=config["policy"],
            env=environment,

            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],

            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],

            clip_range=config["clip_range"],

            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],

            max_grad_norm=config["max_grad_norm"],

            verbose=config["verbose"],
            device=config["device"],
        )

    def train(
        self,
        model: PPO,
        config: dict[str, Any],
        context: Any,
    ) -> None:

        model.learn(
            total_timesteps=config["total_timesteps"],
            reset_num_timesteps=(
                config["reset_num_timesteps"]
            ),
        )

    def save(
        self,
        model: PPO,
        path: str,
    ) -> None:

        model.save(path)

    def load(
        self,
        path: str,
        environment: Any | None = None,
    ) -> PPO:

        return PPO.load(
            path,
            env=environment,
        )


    def predict(
        self,
        model: PPO,
        observation: Any,
        deterministic: bool = True,
    ) -> Any:

        action, _ = model.predict(
            observation,
            deterministic=deterministic,
        )

        return action


def register(
    registry: PluginRegistry,
) -> None:

    registry.register_algorithm(
        StableBaselines3PPOPlugin()
    )