from rl_framework.plugins.algorithm import AlgorithmPlugin
from rl_framework.plugins.environment import EnvironmentPlugin
from rl_framework.plugins.base import (
    PluginMetadata,
    PluginCapabilities,
)
from rl_framework.plugins.registry import PluginRegistry


class DummyAlgorithmPlugin(AlgorithmPlugin):

    @property
    def metadata(self):
        return PluginMetadata(
            name="Dummy Algorithm",
            version="0.1.0",
            description="Plugin usado para testes.",
        )

    @property
    def capabilities(self):
        return PluginCapabilities(
            parallel_instances=True,
            checkpointing=False,
            supported_action_spaces=["Discrete"],
        )

    def config_schema(self):
        return {
            "learning_rate": {
                "type": "float",
                "default": 0.001,
            }
        }

    def create(self, environment, config):
        return {"environment": environment}

    def train(self, model, config, context):
        pass

    def save(self, model, path):
        pass


class DummyEnvironmentPlugin(EnvironmentPlugin):

    @property
    def metadata(self):
        return PluginMetadata(
            name="Dummy Environment",
            version="0.1.0",
        )

    @property
    def capabilities(self):
        return PluginCapabilities(
            parallel_instances=True,
            supported_action_spaces=["Discrete"],
        )

    def config_schema(self):
        return {}

    def create(self, config, instances=1):
        return {
            "instances": instances
        }

    def close(self, environment):
        pass


def test_registry():

    registry = PluginRegistry()

    algorithm = DummyAlgorithmPlugin()
    environment = DummyEnvironmentPlugin()

    registry.register_algorithm(algorithm)
    registry.register_environment(environment)

    assert registry.get_algorithm(
        "Dummy Algorithm"
    ) is algorithm

    assert registry.get_environment(
        "Dummy Environment"
    ) is environment

    assert len(registry.algorithms()) == 1
    assert len(registry.environments()) == 1