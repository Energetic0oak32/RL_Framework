from pathlib import Path

from rl_framework.plugins.discovery import PluginDiscovery
from rl_framework.plugins.registry import PluginRegistry


registry = PluginRegistry()

plugins_directory = Path(__file__).parent / "plugins"

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


print()

print("Algorithms")
print("==========")

for plugin in registry.algorithms():

    print(
        plugin.metadata.name,
        plugin.metadata.version,
    )


print()

print("Environments")
print("============")

for plugin in registry.environments():

    print(
        plugin.metadata.name,
        plugin.metadata.version,
    )


print()

print("Simulators")
print("==========")

for plugin in registry.simulators():

    print(
        plugin.metadata.name,
        plugin.metadata.version,
    )