from pathlib import Path

from rl_framework.core.experiments import Experiment
from rl_framework.plugins.discovery import PluginDiscovery
from rl_framework.plugins.registry import PluginRegistry
from rl_framework.runtime.evaluator import EvaluationRunner


project_root = Path(__file__).parent


# ================================================================
# Discovery
# ================================================================

registry = PluginRegistry()

discovery = PluginDiscovery(
    registry=registry,
    plugins_directory=project_root / "plugins",
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

    output_path=Path(
        "runs/cartpole_ppo/model"
    ),
)


# ================================================================
# Evaluation
# ================================================================

evaluator = EvaluationRunner(
    registry=registry
)


print()
print("Evaluating model")
print("================")


result = evaluator.run(
    experiment=experiment,
    model_path=Path(
        "runs/cartpole_ppo/model.zip"
    ),
    episodes=20,
    deterministic=True,
)


print()
print("Evaluation result")
print("=================")

print(
    f"Episodes:            "
    f"{result.episodes}"
)

print(
    f"Mean reward:         "
    f"{result.mean_reward:.2f}"
)

print(
    f"Reward std:          "
    f"{result.reward_std:.2f}"
)

print(
    f"Mean episode length: "
    f"{result.mean_episode_length:.2f}"
)


print()
print("Episode rewards")
print("===============")

for index, reward in enumerate(
    result.episode_rewards,
    start=1,
):

    print(
        f"{index:02d}: {reward:.2f}"
    )