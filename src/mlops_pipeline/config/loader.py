"""Load and validate ``config/config.yaml``.

Usage::

    from mlops_pipeline.config import load_config
    settings = load_config()                 # validated, raises ConfigError
    settings.config.target                   # value comes from the YAML
    settings.train_path                      # absolute Path

Config location: explicit ``path`` argument > ``MLOPS_CONFIG`` env var >
``config/config.yaml`` found by walking up from the current directory.
The repository root is the parent of the directory holding the config file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .schema import AppConfig
from .validation import find_pending, validate_semantics

CONFIG_ENV_VAR = "MLOPS_CONFIG"
DEFAULT_CONFIG_RELATIVE = Path("config") / "config.yaml"


class ConfigError(Exception):
    """Raised when the configuration cannot be loaded or is invalid."""

    def __init__(self, problems: list[str] | str, source: Path | None = None):
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        self.source = source
        head = f"Invalid configuration ({source})" if source else "Invalid configuration"
        super().__init__(head + ":\n" + "\n".join(f"  - {p}" for p in self.problems))


@dataclass(frozen=True)
class Settings:
    """Validated configuration plus the context needed to resolve paths."""

    config: AppConfig
    config_path: Path
    project_root: Path
    pending: tuple[str, ...]

    def resolve(self, relative: str | Path) -> Path:
        return (self.project_root / relative).resolve()

    @property
    def train_path(self) -> Path:
        return self.resolve(self.config.paths.train_data)

    @property
    def incoming_path(self) -> Path:
        return self.resolve(self.config.paths.incoming_data)


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys (silent overrides)."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def find_config_path(path: str | os.PathLike[str] | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    env = os.environ.get(CONFIG_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    here = Path.cwd().resolve()
    for folder in (here, *here.parents):
        candidate = folder / DEFAULT_CONFIG_RELATIVE
        if candidate.is_file():
            return candidate
    return (Path(__file__).resolve().parents[3] / DEFAULT_CONFIG_RELATIVE)


def read_raw_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.load(fh, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML parse error: {exc}", config_path) from exc
    if not isinstance(raw, dict):
        raise ConfigError("top level of the config must be a mapping", config_path)
    return raw


def load_config(
    path: str | os.PathLike[str] | None = None, *, check_paths: bool = False
) -> Settings:
    """Load, type-check and cross-validate the configuration.

    ``check_paths=True`` additionally requires the raw data files to exist.
    """
    config_path = find_config_path(path)
    raw = read_raw_config(config_path)
    try:
        cfg = AppConfig.model_validate(raw)
    except ValidationError as exc:
        problems = [
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        ]
        raise ConfigError(problems, config_path) from exc

    project_root = config_path.parent.parent
    problems = validate_semantics(cfg, project_root=project_root, check_paths=check_paths)
    if problems:
        raise ConfigError(problems, config_path)

    return Settings(
        config=cfg,
        config_path=config_path,
        project_root=project_root,
        pending=tuple(find_pending(raw)),
    )
