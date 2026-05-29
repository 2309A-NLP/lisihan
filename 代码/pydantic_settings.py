import os
from typing import Any

from pydantic import BaseModel


# Function: Provide lightweight settings loading from environment variables.
class BaseSettings(BaseModel):
    """Small local fallback for pydantic-settings.

    It supports the subset this project uses: defaults, required string fields,
    class Config.env_file, and environment/.env overrides.
    """

    # Function: Initialize instance fields and runtime dependencies.
    def __init__(self, **values: Any):
        config = getattr(self.__class__, "Config", None)
        env_file = getattr(config, "env_file", None)
        case_sensitive = getattr(config, "case_sensitive", False)
        env_values = self._load_env_file(env_file)

        for field_name, field_info in self.__class__.model_fields.items():
            env_key = field_name if case_sensitive else field_name.upper()
            candidates = [field_name, env_key]
            for key in candidates:
                if key in os.environ and field_name not in values:
                    value = self._coerce_env_value(os.environ[key], field_info)
                    if value is not None:
                        values[field_name] = value
                    break
                if key in env_values and field_name not in values:
                    value = self._coerce_env_value(env_values[key], field_info)
                    if value is not None:
                        values[field_name] = value
                    break

        super().__init__(**values)

    # Function: Load key-value pairs from a .env file.
    @staticmethod
    def _load_env_file(env_file: str | None) -> dict[str, str]:
        if not env_file:
            return {}

        path = os.path.abspath(env_file)
        if not os.path.exists(path):
            return {}

        values: dict[str, str] = {}
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
        return values

    # Function: Convert environment strings to the type of the default value.
    @staticmethod
    def _coerce_env_value(value: str, field_info: Any) -> Any:
        if field_info.annotation is bool:
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            if not field_info.is_required():
                return None
        return value
