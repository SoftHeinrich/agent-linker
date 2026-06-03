from llm_sad_sam.llm_client import LLMBackend

import run_ablation


def test_retained_variants_are_exposed():
    variants = run_ablation.available_variants()
    assert "i1" in variants
    assert "i2" in variants
    assert "i3" in variants
    assert "s_linker" in variants
    assert "s_linker11a" in variants
    assert run_ablation.canonical_variant("ilinker1") == "i1"
    assert run_ablation.canonical_variant("ilinker2") == "i2"
    assert run_ablation.canonical_variant("ilinker3") == "i3"
    assert run_ablation.canonical_variant("s_linker1") == "s_linker"


def test_build_linker_smoke(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_SESSION_DIR", str(tmp_path))
    assert run_ablation.build_linker("i1", backend=LLMBackend.CLAUDE).__class__.__name__ == "ILinker1"
    assert run_ablation.build_linker("i3", backend=LLMBackend.CLAUDE).__class__.__name__ == "ILinker3Adapter"
    assert run_ablation.build_linker("s_linker11a", backend=LLMBackend.CLAUDE).__class__.__name__ == "SLinker11a"
