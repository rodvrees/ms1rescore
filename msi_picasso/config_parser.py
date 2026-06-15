"""Cascade configuration for MSI-PICASSO.

Priority (lowest to highest):
  1. package_data/config_default.json  — all defaults
  2. User --config-file (JSON or TOML)
  3. CLI arguments (explicit only; None values never override lower-priority sources)
"""

import json
import sys
from argparse import Namespace
from pathlib import Path

import jsonschema
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


def _normalize_config(d: dict) -> dict:
    """Normalize a top-level config dict.

    The top-level section key is the literal ``"MSI-PICASSO"`` (with a hyphen),
    so it must NOT be hyphen-normalized — only the keys *inside* each section
    are normalized (e.g. ``maldi-raw`` → ``maldi_raw``).
    """
    return {
        k: (_normalize_keys(v) if isinstance(v, dict) else v)
        for k, v in d.items()
    }


def _apply_explicit_overrides(result: dict, overrides: list) -> None:
    """Re-apply explicit non-None values that ``cascade_config`` silently dropped.

    ``cascade_config``'s merge rule is ``elif v or k not in original`` — so a
    falsy-but-explicit scalar (``0``, ``0.0``, ``""``) is NOT applied when the key
    already has a value, leaving the lower-priority (default) value in place.  This
    re-applies user-provided non-None values (later sources win) so an intentional
    ``0`` is honored.  ``None`` still means "unset" and never overrides.  Only keys
    that already exist in the merged section are touched (defaults populate every
    valid key, so this never introduces unknown keys); one level of nested
    sub-sections (e.g. ``maldi_extraction``, ``im2deep``) is handled.
    """
    section = result.get("MSI-PICASSO")
    if not isinstance(section, dict):
        return
    for ov in overrides:
        sec = ov.get("MSI-PICASSO") if isinstance(ov, dict) else None
        if not isinstance(sec, dict):
            continue
        for k, v in sec.items():
            if v is None or k not in section:
                continue
            if isinstance(v, dict) and isinstance(section[k], dict):
                for kk, vv in v.items():
                    if vv is not None and kk in section[k]:
                        section[k][kk] = vv
            elif not isinstance(v, dict):
                section[k] = v


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
        Merged config with top-level key ``"MSI-PICASSO"``.
    """
    schema = json.loads(_SCHEMA.read_text())
    cascade_conf = CascadeConfig(
        validation_schema=schema,
        none_overrides_value=False,
        max_recursion_depth=2,
    )
    cascade_conf.add_json(_DEFAULT)

    # Keep normalized user sources so we can re-apply explicit falsy values that
    # cascade_config drops (see _apply_explicit_overrides).
    user_overrides: list = []
    for config in (configurations or []):
        if isinstance(config, dict):
            nc = _normalize_config(config)
            cascade_conf.add_dict(nc)
            user_overrides.append(nc)
        elif isinstance(config, (str, Path)):
            p = Path(config)
            if p.suffix.lower() == ".json":
                nc = _normalize_config(json.loads(p.read_text()))
            elif p.suffix.lower() in (".toml", ".tml"):
                nc = _normalize_config(tomllib.loads(p.read_text()))
            else:
                raise ValueError(
                    f"Unsupported config file format: {p.suffix!r}. "
                    "Use .json or .toml."
                )
            cascade_conf.add_dict(nc)
            user_overrides.append(nc)
        elif isinstance(config, Namespace):
            cascade_conf.add_namespace(config, subkey="MSI-PICASSO")
            user_overrides.append({"MSI-PICASSO": dict(vars(config))})
        else:
            raise TypeError(
                f"Unsupported config source type: {type(config).__name__}. "
                "Expected a file path, argparse.Namespace, or dict."
            )

    result = cascade_conf.parse()
    # cascade_config drops explicit falsy scalars (e.g. an intentional 0); re-apply
    # them, then re-validate so an out-of-range explicit value (e.g. matching_ppm=0,
    # forbidden by the schema's exclusiveMinimum) raises a clear error instead of
    # being silently masked by the default.
    _apply_explicit_overrides(result, user_overrides)
    jsonschema.validate(result, schema)
    return result
