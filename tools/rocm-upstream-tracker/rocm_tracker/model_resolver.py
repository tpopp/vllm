from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelMaps:
    arch_to_module: dict[str, str]
    module_to_archs: dict[str, list[str]]

    def architectures_for_file(self, path: str) -> list[tuple[str, str, str]]:
        """Return (architecture, module, impact_level) tuples for a changed file."""
        results: list[tuple[str, str, str]] = []
        if path == "vllm/model_executor/models/registry.py":
            return results

        model_match = re.match(
            r"vllm/model_executor/models/(?P<module>[a-zA-Z0-9_]+)\.py$",
            path,
        )
        if model_match:
            module = model_match.group("module")
            for arch in self.module_to_archs.get(module, []):
                results.append((arch, module, "definite"))
            return results

        shared_prefixes = (
            "vllm/model_executor/layers/fused_moe/",
            "vllm/v1/attention/backends/",
            "vllm/model_executor/layers/quantization/",
            "vllm/model_executor/kernels/",
        )
        if path.startswith(shared_prefixes):
            for arch, module in self.arch_to_module.items():
                results.append((arch, module, "possible"))
        return results


def _extract_model_dicts(source: str) -> dict[str, tuple[str, str]]:
    tree = ast.parse(source)
    mappings: dict[str, tuple[str, str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if not target.id.startswith("_") or "MODEL" not in target.id:
                continue
            if not isinstance(node.value, ast.Dict):
                continue
            for key, value in zip(node.value.keys, node.value.values):
                if not isinstance(key, ast.Constant) or not isinstance(
                    value, ast.Tuple
                ):
                    continue
                if len(value.elts) < 2:
                    continue
                arch = str(key.value)
                if not isinstance(value.elts[0], ast.Constant):
                    continue
                module = str(value.elts[0].value)
                mappings[arch] = (module, arch)
    return mappings


def load_model_maps(registry_path: Path) -> ModelMaps:
    source = registry_path.read_text(encoding="utf-8")
    arch_to_module_raw = _extract_model_dicts(source)
    arch_to_module = {arch: module for arch, (module, _) in arch_to_module_raw.items()}
    module_to_archs: dict[str, list[str]] = {}
    for arch, module in arch_to_module.items():
        module_to_archs.setdefault(module, []).append(arch)
    return ModelMaps(arch_to_module=arch_to_module, module_to_archs=module_to_archs)


def impacts_from_files(
    maps: ModelMaps,
    changed_files: list[str],
) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    impacts: list[dict[str, str]] = []
    for path in changed_files:
        for arch, module, level in maps.architectures_for_file(path):
            key = (arch, level)
            if key in seen:
                continue
            seen.add(key)
            impacts.append(
                {
                    "architecture": arch,
                    "module_name": module,
                    "impact": level,
                    "rationale": f"Changed path: {path}",
                }
            )
    return impacts


def parse_registry_arch_changes(diff_text: str) -> list[str]:
    added: list[str] = []
    for line in diff_text.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        match = re.search(r'"([A-Za-z0-9_]+)":\s*\(', line)
        if match:
            added.append(match.group(1))
    return added
