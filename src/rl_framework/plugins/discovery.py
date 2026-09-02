from __future__ import annotations

import importlib.util

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from .registry import PluginRegistry


@dataclass
class PluginLoadResult:
    """
    Resultado da tentativa de carregar um plugin.
    """

    name: str
    path: Path
    success: bool
    error: str | None = None


class PluginDiscovery:
    """
    Descobre e carrega plugins externos a partir de um diretório.

    Convenção:

        plugins/
            meu_plugin/
                plugin.py

    Cada plugin.py deve expor:

        def register(registry: PluginRegistry) -> None:
            ...
    """

    def __init__(
        self,
        registry: PluginRegistry,
        plugins_directory: str | Path,
    ) -> None:

        self.registry = registry
        self.plugins_directory = Path(plugins_directory)

    def discover(self) -> list[PluginLoadResult]:
        """
        Procura plugins no diretório configurado e tenta carregá-los.
        """

        results: list[PluginLoadResult] = []

        if not self.plugins_directory.exists():
            return results

        for directory in sorted(self.plugins_directory.iterdir()):

            if not directory.is_dir():
                continue

            plugin_file = directory / "plugin.py"

            if not plugin_file.exists():
                continue

            result = self._load_plugin(
                name=directory.name,
                plugin_file=plugin_file,
            )

            results.append(result)

        return results

    def _load_plugin(
        self,
        name: str,
        plugin_file: Path,
    ) -> PluginLoadResult:

        try:

            module = self._import_module(
                name=name,
                plugin_file=plugin_file,
            )

            register = getattr(
                module,
                "register",
                None,
            )

            if register is None or not callable(register):
                raise RuntimeError(
                    "Plugin does not expose a callable "
                    "'register(registry)' function."
                )

            register(self.registry)

            return PluginLoadResult(
                name=name,
                path=plugin_file,
                success=True,
            )

        except Exception as exc:

            return PluginLoadResult(
                name=name,
                path=plugin_file,
                success=False,
                error=str(exc),
            )

    @staticmethod
    def _import_module(
        name: str,
        plugin_file: Path,
    ) -> ModuleType:

        module_name = f"rl_framework_external_plugin_{name}"

        spec = importlib.util.spec_from_file_location(
            module_name,
            plugin_file,
        )

        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not create import specification "
                f"for plugin '{name}'."
            )

        module = importlib.util.module_from_spec(spec)

        spec.loader.exec_module(module)

        return module