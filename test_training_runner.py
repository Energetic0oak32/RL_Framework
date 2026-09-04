from pathlib import Path

from rl_framework.core.experiments import Experiment
from rl_framework.plugins.discovery import PluginDiscovery
from rl_framework.plugins.registry import PluginRegistry
from rl_framework.runtime.runner import TrainingRunner


project_root = Path(__file__).parent

plugins_directory = project_root / "plugins"


# ================================================================
# Plugin discovery
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

    status = "OK" if result.success else "FAILED"

    print(
        f"{status:6} | "
        f"{result.name:20} | "
        f"{result.error or ''}"
    )


# ================================================================
# Experiment
# ================================================================

experiment = Experiment(
    algorithm="Test Algorithm",
    environment="Test Environment",
    instances=4,

    algorithm_config={},
    
    environment_config={},

    simulator_config={},

    output_path=Path(
        "runs/test_training/model"
    ),
)


# ================================================================
# Training
# ================================================================

runner = TrainingRunner(
    registry=registry
)

print()
print("Starting experiment")
print("===================")

runner.run(
    experiment
)

print()
print("Experiment finished")