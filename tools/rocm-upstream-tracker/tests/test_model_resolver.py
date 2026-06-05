from pathlib import Path

from rocm_tracker.model_resolver import impacts_from_files, load_model_maps


def test_llama_module_maps_to_llama_architectures():
    registry = Path(__file__).resolve().parents[3] / "vllm/model_executor/models/registry.py"
    maps = load_model_maps(registry)
    impacts = impacts_from_files(
        maps,
        ["vllm/model_executor/models/llama.py"],
    )
    archs = {i["architecture"] for i in impacts}
    assert "LlamaForCausalLM" in archs
    assert all(i["impact"] == "definite" for i in impacts)


def test_shared_moe_path_is_possible():
    registry = Path(__file__).resolve().parents[3] / "vllm/model_executor/models/registry.py"
    maps = load_model_maps(registry)
    impacts = impacts_from_files(
        maps,
        ["vllm/model_executor/layers/fused_moe/layer.py"],
    )
    assert impacts
    assert any(i["impact"] == "possible" for i in impacts)
