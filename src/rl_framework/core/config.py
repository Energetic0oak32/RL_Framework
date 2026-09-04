from __future__ import annotations

from typing import Any

from rl_framework.plugins.base import BasePlugin


class ConfigValidationError(ValueError):
    """
    Erro levantado quando a configuração de um plugin é inválida.
    """

    pass


def validate_plugin_config(
    plugin: BasePlugin,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Valida uma configuração usando o config_schema() do plugin.

    Também aplica valores default quando eles não foram
    explicitamente informados.
    """

    schema = plugin.config_schema()

    resolved: dict[str, Any] = {}

    # ============================================================
    # Unknown parameters
    # ============================================================

    unknown_fields = set(config) - set(schema)

    if unknown_fields:
        names = ", ".join(
            sorted(unknown_fields)
        )

        raise ConfigValidationError(
            f"Unknown configuration field(s) for "
            f"'{plugin.metadata.name}': {names}"
        )

    # ============================================================
    # Schema fields
    # ============================================================

    for name, field_schema in schema.items():

        if name in config:
            value = config[name]

        elif "default" in field_schema:
            value = field_schema["default"]

        elif field_schema.get("required", False):
            raise ConfigValidationError(
                f"Missing required configuration field "
                f"'{name}' for plugin "
                f"'{plugin.metadata.name}'."
            )

        else:
            continue

        expected_type = field_schema.get("type")

        if expected_type is not None:
            _validate_type(
                plugin_name=plugin.metadata.name,
                field_name=name,
                value=value,
                expected_type=expected_type,
            )

        choices = field_schema.get("choices")

        if choices is not None:
            if value not in choices:
                raise ConfigValidationError(
                    f"Invalid value for '{name}' in plugin "
                    f"'{plugin.metadata.name}': {value!r}. "
                    f"Expected one of: {choices}."
                )

        minimum = field_schema.get("min")

        if minimum is not None:
            if value < minimum:
                raise ConfigValidationError(
                    f"Configuration field '{name}' for plugin "
                    f"'{plugin.metadata.name}' must be >= "
                    f"{minimum}."
                )

        maximum = field_schema.get("max")

        if maximum is not None:
            if value > maximum:
                raise ConfigValidationError(
                    f"Configuration field '{name}' for plugin "
                    f"'{plugin.metadata.name}' must be <= "
                    f"{maximum}."
                )

        resolved[name] = value

    return resolved


def _validate_type(
    plugin_name: str,
    field_name: str,
    value: Any,
    expected_type: str,
) -> None:

    valid = False

    if expected_type == "int":
        valid = (
            isinstance(value, int)
            and not isinstance(value, bool)
        )

    elif expected_type == "float":
        valid = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        )

    elif expected_type in {
        "str",
        "string",
    }:
        valid = isinstance(value, str)

    elif expected_type == "bool":
        valid = isinstance(value, bool)

    elif expected_type == "list":
        valid = isinstance(value, list)

    elif expected_type == "dict":
        valid = isinstance(value, dict)

    else:
        raise ConfigValidationError(
            f"Plugin '{plugin_name}' declares unsupported "
            f"configuration type '{expected_type}' "
            f"for field '{field_name}'."
        )

    if not valid:
        raise ConfigValidationError(
            f"Invalid type for '{field_name}' in plugin "
            f"'{plugin_name}'. Expected {expected_type}, "
            f"received {type(value).__name__}."
        )