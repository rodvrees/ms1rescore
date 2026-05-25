"""Cascade configuration for ms1rescore.

Priority (lowest to highest):
  1. package_data/config_default.json  — all defaults
  2. User --config-file (JSON or TOML)
  3. CLI arguments (explicit only; None values never override lower-priority sources)
"""

import json
import sys
from argparse import Namespace
from pathlib import Path

from cascade_config import CascadeConfig

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


_SCHEMA = Path(__file__).parent / "package_data" / "config_schema.json"
_DEFAULT = Path(__file__).parent / "package_data" / "config_default.json"


def _normalize_keys(d: dict) -> dict:
    """Recursively replace hyphens with underscores in dict keys.

    Allows TOML/JSON config files to use either ``maldi-raw`` or ``maldi_raw``
    — both map to the same schema key.
    """
    return {
        k.replace("-", "_"): (_normalize_keys(v) if isinstance(v, dict) else v)
        for k, v in d.items()
    }


def parse_configurations(configurations=None) -> dict:
    """Merge config sources and return the resolved config dict.

    Parameters
    ----------
    configurations
        Ordered list of config sources to layer on top of defaults.
        Each item may be:
        - a file path (str or Path) to a JSON or TOML config file
        - an argparse.Namespace (CLI args)
        - a plain dict

    Returns
    -------
    dict
        Merged config with top-level key ``"ms1rescore"``.
    """
    cascade_conf = CascadeConfig(
        validation_schema=json.loads(_SCHEMA.read_text()),
        none_overrides_value=False,
        max_recursion_depth=2,
    )
    cascade_conf.add_json(_DEFAULT)

    for config in (configurations or []):
        if isinstance(config, dict):
            cascade_conf.add_dict(_normalize_keys(config))
        elif isinstance(config, (str, Path)):
            p = Path(config)
            if p.suffix.lower() == ".json":
                cascade_conf.add_dict(_normalize_keys(json.loads(p.read_text())))
            elif p.suffix.lower() in (".toml", ".tml"):
                cascade_conf.add_dict(_normalize_keys(tomllib.loads(p.read_text())))
            else:
                raise ValueError(
                    f"Unsupported config file format: {p.suffix!r}. "
                    "Use .json or .toml."
                )
        elif isinstance(config, Namespace):
            cascade_conf.add_namespace(config, subkey="ms1rescore")
        else:
            raise TypeError(
                f"Unsupported config source type: {type(config).__name__}. "
                "Expected a file path, argparse.Namespace, or dict."
            )

    return cascade_conf.parse()
