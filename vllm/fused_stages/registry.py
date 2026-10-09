# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Provider discovery.

Two sources, one contract:

* Out-of-tree providers: the ``vllm.fused_stages`` entry point group. Each
  entry point resolves to a ``StageProvider`` subclass or a zero-argument
  factory returning an instance. ``VLLM_PLUGINS`` filters by entry point name
  like every other vLLM plugin group.
* In-tree providers: ``IN_TREE_PROVIDERS`` maps a provider id to a lazy
  ``"module:attr"`` path. In-tree providers import only
  ``vllm.fused_stages.api`` and live under ``vllm/fused_stages/providers/``,
  so moving one out of tree is a file move plus an entry point.
"""

from __future__ import annotations

import importlib
from functools import cache
from importlib.metadata import entry_points

from vllm.fused_stages.api import (
    ENTRY_POINT_GROUP,
    INTERFACE_MAJOR,
    REFERENCE,
    StageProvider,
)
from vllm.logger import init_logger
from vllm.plugins import load_plugins_by_group

logger = init_logger(__name__)

IN_TREE_PROVIDERS: dict[str, str] = {
    # Phase 1 of the plan in docs/design/fused_stages.md lands a reference
    # provider in tree, e.g.:
    # "rocm_mono_dsv41": "vllm.fused_stages.providers.rocm_mono_dsv41:Provider",
}

_TEST_PROVIDERS: dict[str, StageProvider] = {}


def register_provider_for_testing(provider: StageProvider) -> None:
    """Register a provider instance directly (tests, notebooks)."""
    _TEST_PROVIDERS[provider.provider_id] = provider
    discover_providers.cache_clear()


def _instantiate(obj: object, origin: str) -> StageProvider | None:
    try:
        provider = obj() if callable(obj) else obj  # class or factory
    except Exception:
        logger.exception("Failed to instantiate fused-stage provider %s", origin)
        return None
    if not isinstance(provider, StageProvider):
        logger.error("%s is not a StageProvider; ignored", origin)
        return None
    if provider.provider_id == REFERENCE:
        logger.error("%s uses the reserved provider id %r", origin, REFERENCE)
        return None
    if provider.interface_major != INTERFACE_MAJOR:
        logger.warning(
            "Fused-stage provider %s targets interface %d, this vLLM has %d; ignored.",
            provider.provider_id,
            provider.interface_major,
            INTERFACE_MAJOR,
        )
        return None
    return provider


@cache
def discover_providers() -> dict[str, StageProvider]:
    """All usable providers by id (in-tree first, then plugins)."""
    found: dict[str, StageProvider] = {}
    for pid, path in IN_TREE_PROVIDERS.items():
        module, _, attr = path.partition(":")
        try:
            obj = getattr(importlib.import_module(module), attr)
        except ImportError as err:  # optional deps (e.g. FlyDSL) missing
            logger.debug("In-tree fused-stage provider %s unavailable: %s", pid, err)
            continue
        if (provider := _instantiate(obj, path)) is not None:
            found[provider.provider_id] = provider

    for name, obj in load_plugins_by_group(ENTRY_POINT_GROUP).items():
        provider = _instantiate(obj, f"entry point {name}")
        if provider is None:
            continue
        if provider.provider_id in found:
            logger.warning(
                "Fused-stage provider %s (entry point %s) shadows an existing "
                "provider with the same id; ignored.",
                provider.provider_id,
                name,
            )
            continue
        found[provider.provider_id] = provider

    found.update(_TEST_PROVIDERS)
    return found


def installed_provider_versions() -> dict[str, str]:
    """Entry point name -> distribution version, without importing plugins.
    Used for the compile/graph cache hash."""
    versions: dict[str, str] = {}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        dist = getattr(ep, "dist", None)
        versions[ep.name] = dist.version if dist is not None else "unknown"
    return versions
