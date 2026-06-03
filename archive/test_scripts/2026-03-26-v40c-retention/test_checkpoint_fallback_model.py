#!/usr/bin/env python3
"""Regression tests for checkpoint fallback model selection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.linkers.experimental.s_linker11 import SLinker11
from llm_sad_sam.linkers.experimental.s_linker11a import SLinker11a


def _clear_llm_env(monkeypatch):
    for key in (
        "CLAUDE_MODEL",
        "OPENAI_MODEL_NAME",
        "CHECKPOINT_FALLBACK",
        "CHECKPOINT_FALLBACK_MODEL",
        "LLM_SESSION_DIR",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LLM_SESSION_DIR", "/tmp/llm-sad-sam-test")


def test_slinker11_checkpoint_sonnet_fallback(monkeypatch):
    _clear_llm_env(monkeypatch)

    linker = SLinker11(
        backend=LLMBackend.CHECKPOINT,
        checkpoint_fallback_model="sonnet",
    )

    assert linker.llm.backend == LLMBackend.CHECKPOINT
    assert linker.llm._checkpoint_fallback == LLMBackend.CLAUDE
    assert linker.llm.claude_model == "sonnet"
    assert linker.llm.describe_backend() == "checkpoint -> claude (sonnet)"


def test_slinker11a_checkpoint_gpt_fallback(monkeypatch):
    _clear_llm_env(monkeypatch)

    linker = SLinker11a(
        backend=LLMBackend.CHECKPOINT,
        checkpoint_fallback_model="gpt",
    )

    assert linker.llm.backend == LLMBackend.CHECKPOINT
    assert linker.llm._checkpoint_fallback == LLMBackend.OPENAI
    assert linker.llm.openai_model == "gpt-5.2"
    assert linker.llm.describe_backend() == "checkpoint -> openai (gpt-5.2)"


def test_checkpoint_fallback_model_env_is_honored(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("CHECKPOINT_FALLBACK_MODEL", "gpt")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "gpt-5.2")

    linker = SLinker11(backend=LLMBackend.CHECKPOINT)

    assert linker.llm._checkpoint_fallback == LLMBackend.OPENAI
    assert linker.llm.openai_model == "gpt-5.2"
