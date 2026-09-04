from __future__ import annotations

from pathlib import Path
from typing import Any

from rl_framework.core.compatibility import (
    validate_plugin_compatibility,
)
from rl_framework.core.config import (
    validate_plugin_config,
)
from rl_framework.core.evaluation import (
    EvaluationResult,
)
from rl_framework.core.experiments import Experiment
from rl_framework.plugins.registry import PluginRegistry


class EvaluationRunner:
    """
    Avalia um modelo treinado em um EnvironmentPlugin.

    Nesta primeira implementação, avaliação é feita
    usando uma única instância do ambiente.
    """

    def __init__(
        self,
        registry: PluginRegistry,
    ) -> None:

        self.registry = registry

    def run(
        self,
        experiment: Experiment,
        model_path: str | Path,
        episodes: int = 10,
        deterministic: bool = True,
    ) -> EvaluationResult:

        if episodes < 1:
            raise ValueError(
                "Evaluation episodes must be at least 1."
            )

        if experiment.instances != 1:
            raise ValueError(
                "Evaluation currently supports only "
                "one environment instance."
            )

        # ========================================================
        # Resolve plugins
        # ========================================================

        algorithm_plugin = self.registry.get_algorithm(
            experiment.algorithm
        )

        environment_plugin = self.registry.get_environment(
            experiment.environment
        )

        # ========================================================
        # Config
        # ========================================================

        algorithm_config = validate_plugin_config(
            plugin=algorithm_plugin,
            config=experiment.algorithm_config,
        )

        environment_config = validate_plugin_config(
            plugin=environment_plugin,
            config=experiment.environment_config,
        )

        # algorithm_config is validated intentionally even though
        # evaluation does not currently use every training option.
        _ = algorithm_config

        # ========================================================
        # Simulator
        # ========================================================

        simulator_plugin = None
        simulator_config: dict[str, Any] = {}

        required_simulator = (
            environment_plugin.required_simulator
        )

        if required_simulator is not None:

            simulator_plugin = self.registry.get_simulator(
                required_simulator
            )

            simulator_config = validate_plugin_config(
                plugin=simulator_plugin,
                config=experiment.simulator_config,
            )

        # ========================================================
        # Compatibility
        # ========================================================

        validate_plugin_compatibility(
            algorithm=algorithm_plugin,
            environment=environment_plugin,
            environment_config=environment_config,
            instances=1,
            simulator=simulator_plugin,
        )

        # ========================================================
        # Runtime
        # ========================================================

        simulator = None
        environment = None

        rewards: list[float] = []
        lengths: list[int] = []

        try:

            if simulator_plugin is not None:

                simulator = simulator_plugin.launch(
                    config=simulator_config,
                    instances=1,
                )

            environment = environment_plugin.create(
                config=environment_config,
                simulator=simulator,
                instances=1,
            )

            # ====================================================
            # Load model
            # ====================================================

            model = algorithm_plugin.load(
                path=str(model_path),
                environment=environment,
            )

            # ====================================================
            # Episodes
            # ====================================================

            for _ in range(episodes):

                observation, _ = environment.reset()

                terminated = False
                truncated = False

                episode_reward = 0.0
                episode_length = 0

                while not (
                    terminated
                    or truncated
                ):

                    action = algorithm_plugin.predict(
                        model=model,
                        observation=observation,
                        deterministic=deterministic,
                    )

                    (
                        observation,
                        reward,
                        terminated,
                        truncated,
                        _,
                    ) = environment.step(
                        action
                    )

                    episode_reward += float(
                        reward
                    )

                    episode_length += 1

                rewards.append(
                    episode_reward
                )

                lengths.append(
                    episode_length
                )

            return EvaluationResult(
                episode_rewards=rewards,
                episode_lengths=lengths,
            )

        finally:

            try:

                if environment is not None:

                    environment_plugin.close(
                        environment
                    )

            finally:

                if (
                    simulator is not None
                    and simulator_plugin is not None
                ):

                    simulator_plugin.close(
                        simulator
                    )