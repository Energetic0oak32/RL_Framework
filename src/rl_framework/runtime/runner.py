from __future__ import annotations

from typing import Any

from rl_framework.core.compatibility import (
    validate_plugin_compatibility,
)
from rl_framework.core.config import (
    validate_plugin_config,
)
from rl_framework.core.experiments import Experiment
from rl_framework.plugins.registry import PluginRegistry

from .context import TrainingContext


class TrainingRunner:
    """
    Responsável por orquestrar a execução
    de um experimento.
    """

    def __init__(
        self,
        registry: PluginRegistry,
    ) -> None:
        self.registry = registry

    def run(
        self,
        experiment: Experiment,
    ) -> Any:

        # ========================================================
        # Resolve algorithm and environment
        # ========================================================

        algorithm_plugin = self.registry.get_algorithm(
            experiment.algorithm
        )

        environment_plugin = self.registry.get_environment(
            experiment.environment
        )

        # ========================================================
        # Validate algorithm/environment configs
        # ========================================================

        algorithm_config = validate_plugin_config(
            plugin=algorithm_plugin,
            config=experiment.algorithm_config,
        )

        environment_config = validate_plugin_config(
            plugin=environment_plugin,
            config=experiment.environment_config,
        )

        # ========================================================
        # Resolve simulator
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
        # Validate plugin compatibility
        # ========================================================

        validate_plugin_compatibility(
            algorithm=algorithm_plugin,
            environment=environment_plugin,
            environment_config=environment_config,
            instances=experiment.instances,
            simulator=simulator_plugin,
        )

        # ========================================================
        # Runtime preparation
        # ========================================================

        simulator = None
        environment = None
        model = None

        experiment.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        context = TrainingContext(
            experiment=experiment,
            output_path=experiment.output_path,
        )

        try:

            # ====================================================
            # Simulator
            # ====================================================

            if simulator_plugin is not None:

                simulator = simulator_plugin.launch(
                    config=simulator_config,
                    instances=experiment.instances,
                )

            # ====================================================
            # Environment
            # ====================================================

            environment = environment_plugin.create(
                config=environment_config,
                simulator=simulator,
                instances=experiment.instances,
            )

            # ====================================================
            # Algorithm
            # ====================================================

            model = algorithm_plugin.create(
                environment=environment,
                config=algorithm_config,
            )

            # ====================================================
            # Training
            # ====================================================

            algorithm_plugin.train(
                model=model,
                config=algorithm_config,
                context=context,
            )

            # ====================================================
            # Save
            # ====================================================

            algorithm_plugin.save(
                model=model,
                path=str(
                    experiment.output_path
                ),
            )

            return model

        finally:

            # ====================================================
            # Cleanup
            # ====================================================

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