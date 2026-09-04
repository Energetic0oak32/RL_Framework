from __future__ import annotations

from rl_framework.plugins.algorithm import AlgorithmPlugin
from rl_framework.plugins.environment import EnvironmentPlugin
from rl_framework.plugins.simulator import SimulatorPlugin


class CompatibilityError(ValueError):
    """
    Erro levantado quando plugins selecionados
    não podem trabalhar juntos.
    """

    pass


def validate_plugin_compatibility(
    algorithm: AlgorithmPlugin,
    environment: EnvironmentPlugin,
    environment_config: dict,
    instances: int,
    simulator: SimulatorPlugin | None = None,
) -> None:
    """
    Valida se algoritmo, ambiente e simulador podem
    participar do mesmo experimento.
    """

    environment_spec = environment.environment_spec(
        environment_config
    )

    algorithm_capabilities = algorithm.capabilities
    environment_capabilities = environment.capabilities

    # ============================================================
    # Action space
    # ============================================================

    supported_action_spaces = (
        algorithm_capabilities.supported_action_spaces
    )

    if (
        environment_spec.action_space
        not in supported_action_spaces
    ):
        raise CompatibilityError(
            f"Algorithm '{algorithm.metadata.name}' does not "
            f"support action space "
            f"'{environment_spec.action_space}' required by "
            f"environment '{environment.metadata.name}'. "
            f"Supported action spaces: "
            f"{supported_action_spaces}."
        )

    # ============================================================
    # Observation space
    # ============================================================

    supported_observation_spaces = (
        algorithm_capabilities.supported_observation_spaces
    )

    if (
        environment_spec.observation_space
        not in supported_observation_spaces
    ):
        raise CompatibilityError(
            f"Algorithm '{algorithm.metadata.name}' does not "
            f"support observation space "
            f"'{environment_spec.observation_space}' required by "
            f"environment '{environment.metadata.name}'. "
            f"Supported observation spaces: "
            f"{supported_observation_spaces}."
        )

    # ============================================================
    # Parallel algorithm execution
    # ============================================================

    if (
        instances > 1
        and not algorithm_capabilities.parallel_instances
    ):
        raise CompatibilityError(
            f"Algorithm '{algorithm.metadata.name}' does not "
            f"support parallel instances, but experiment "
            f"requested {instances} instances."
        )

    # ============================================================
    # Parallel environment execution
    # ============================================================

    if (
        instances > 1
        and not environment_capabilities.parallel_instances
    ):
        raise CompatibilityError(
            f"Environment '{environment.metadata.name}' does not "
            f"support parallel instances, but experiment "
            f"requested {instances} instances."
        )

    # ============================================================
    # Simulator requirement
    # ============================================================

    required_simulator = environment.required_simulator

    if required_simulator is not None:

        if simulator is None:
            raise CompatibilityError(
                f"Environment '{environment.metadata.name}' "
                f"requires simulator '{required_simulator}', "
                f"but no simulator was provided."
            )

        if simulator.metadata.name != required_simulator:
            raise CompatibilityError(
                f"Environment '{environment.metadata.name}' "
                f"requires simulator '{required_simulator}', "
                f"but '{simulator.metadata.name}' was provided."
            )

    # ============================================================
    # Parallel simulator execution
    # ============================================================

    if (
        instances > 1
        and simulator is not None
        and not simulator.capabilities.parallel_instances
    ):
        raise CompatibilityError(
            f"Simulator '{simulator.metadata.name}' does not "
            f"support parallel instances, but experiment "
            f"requested {instances} instances."
        )