#!/usr/bin/env python3
"""Run s_linker19 ablation through a local proxy using the OpenAI backend.

This wrapper intentionally leaves `run_ablation.py` and the linker code
unchanged. It sets process environment defaults, registers s_linker19 in the
runner registry at startup, then delegates to `run_ablation.main()`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).parent
DEFAULT_DATASET = "mediastore"
DEFAULT_RESULTS_DIR = "results/s19_proxy_ablation"
PROXY_URL = os.environ.get("S19_HTTP_PROXY_URL", "http://127.0.0.1:8118")


def load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


load_dotenv()

for key in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
):
    os.environ[key] = PROXY_URL

for key in ("NO_PROXY", "no_proxy"):
    os.environ[key] = ""

os.environ.setdefault("LLM_BACKEND", "openai")
os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.4")
os.environ.setdefault("LLM_SESSION_DIR", "results/llm_sessions")

import run_ablation  # noqa: E402  (env defaults must be set before import)


def register_s_linker19() -> None:
    """Add s_linker19 to run_ablation's in-memory registry."""
    name = "s_linker19"
    if name not in run_ablation.CANONICAL_VARIANTS:
        run_ablation.CANONICAL_VARIANTS.append(name)
    run_ablation.VARIANT_SPECS[name] = {
        "aliases": ("s19",),
        "module": "llm_sad_sam.linkers.experimental.s_linker19",
        "class_name": "SLinker19",
        "description": "S-Linker19 paper variant; wrapper-registered for proxy OpenAI runs.",
        "canonical": False,
        "experimental": True,
    }
    run_ablation.VARIANTS[name] = {
        "canonical": name,
        "description": run_ablation.VARIANT_SPECS[name]["description"],
    }
    run_ablation.VARIANTS["s19"] = {
        "canonical": name,
        "description": "Alias for s_linker19",
    }


def _has_option(argv: list[str], *names: str) -> bool:
    return any(arg in names or any(arg.startswith(f"{name}=") for name in names) for arg in argv)


def _with_defaults(argv: list[str]) -> list[str]:
    if "--list-variants" in argv or "--list-datasets" in argv:
        return argv
    args = list(argv)
    if not _has_option(args, "--variants"):
        args = ["--variants", "s_linker19", *args]
    if not _has_option(args, "--datasets"):
        args = ["--datasets", DEFAULT_DATASET, *args]
    if not _has_option(args, "--results-dir"):
        args = ["--results-dir", DEFAULT_RESULTS_DIR, *args]
    return args


def main(argv: list[str] | None = None) -> int:
    register_s_linker19()
    args = _with_defaults(list(sys.argv[1:] if argv is None else argv))
    print(f"Proxy: {PROXY_URL}")
    print(f"LLM_BACKEND: {os.environ['LLM_BACKEND']}")
    print(f"OPENAI_MODEL_NAME: {os.environ['OPENAI_MODEL_NAME']}")
    print(f"Datasets default: {DEFAULT_DATASET}")
    print(f"Results dir default: {DEFAULT_RESULTS_DIR}")
    return run_ablation.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
