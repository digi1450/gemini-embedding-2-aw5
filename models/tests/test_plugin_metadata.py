from pathlib import Path

import yaml


PLUGIN_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(relative_path: str):
    return yaml.safe_load((PLUGIN_ROOT / relative_path).read_text(encoding="utf-8"))


def test_manifest_registers_embedding_only_provider():
    manifest = _load_yaml("manifest.yaml")
    assert manifest["name"] == "gemini_embedding_2_aw5"
    assert manifest["plugins"]["models"] == [
        "provider/gemini_embedding_2_aw5.yaml"
    ]
    assert manifest["resource"]["permission"]["model"]["text_embedding"] is True
    assert manifest["resource"]["permission"]["model"]["llm"] is False


def test_provider_lists_only_stable_embedding_model():
    provider = _load_yaml("provider/gemini_embedding_2_aw5.yaml")
    position = _load_yaml("models/text_embedding/_position.yaml")
    model = _load_yaml("models/text_embedding/gemini-embedding-2.yaml")

    assert provider["provider"] == "gemini_embedding_2_aw5"
    assert provider["supported_model_types"] == ["text-embedding"]
    assert position == ["gemini-embedding-2"]
    assert model["model"] == "gemini-embedding-2"
    assert model["features"] == ["vision"]
