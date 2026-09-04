from pathlib import Path

from rl_framework.core.experiments import Experiment
from rl_framework.plugins.discovery import PluginDiscovery
from rl_framework.plugins.registry import PluginRegistry
from rl_framework.runtime.runner import TrainingRunner


project_root = Path(__file__).parent

plugins_directory = (
    project_root
    / "plugins"
)


# ================================================================
# Discovery
# ================================================================

registry = PluginRegistry()

discovery = PluginDiscovery(
    registry=registry,
    plugins_directory=plugins_directory,
)

results = discovery.discover()


print()

print("Plugin loading results")
print("======================")

for result in results:

    status = (
        "OK"
        if result.success
        else "FAILED"
    )

    print(
        f"{status:6} | "
        f"{result.name:20} | "
        f"{result.error or ''}"
    )


# ================================================================
# Experiment
# ================================================================

experiment = Experiment(
    algorithm="PPO",
    environment="CartPole",

    instances=1,

    algorithm_config={
        "total_timesteps": 10_000,
    },

    environment_config={},
    simulator_config={},

    output_path=Path(
        "runs/cartpole_ppo/model"
    ),
)


# ================================================================
# Run
# ================================================================

runner = TrainingRunner(
    registry=registry,
)


print()
print("Starting real PPO experiment")
print("============================")


runner.run(
    experiment
)


print()
print("Experiment finished")
print("===================")


model_path = Path(
    "runs/cartpole_ppo/model.zip"
)


if model_path.exists():

    print(
        f"Model created successfully: "
        f"{model_path}"
    )

else:

    raise RuntimeError(
        f"Expected model was not created: "
        f"{model_path}"
    )