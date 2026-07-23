#!/usr/bin/env python3
"""Seed an LLM checkpoint cache from tracked S21/S23 call-trace JSON files.

Only successful trace entries with a complete prompt and response are imported.
The cache key is the same prompt SHA-256 used by ``LLMClient``.  Use traces from
the same model/configuration as the replay; this tool deliberately does not claim
that prompts from different provider settings are interchangeable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("traces", nargs="+", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "approach" / "src"))
    from diskcache import Cache
    from llm_sad_sam.llm_client import LLMClient

    imported = skipped = 0
    with Cache(str(args.checkpoint_dir)) as cache:
        for trace_path in args.traces:
            entries = json.loads(trace_path.read_text())
            if not isinstance(entries, list):
                raise ValueError(f"{trace_path} is not a call-trace list")
            for entry in entries:
                prompt = entry.get("prompt")
                response = entry.get("response_text")
                if not entry.get("success") or not isinstance(prompt, str) or not isinstance(response, str):
                    skipped += 1
                    continue
                key = LLMClient._prompt_hash(prompt)
                cache[key] = {
                    "text": response,
                    "success": True,
                    "error": None,
                    "model": entry.get("model"),
                    "latency_ms": entry.get("latency_ms"),
                    "token_usage": entry.get("token_usage"),
                }
                imported += 1
    print(f"Imported {imported} successful responses; skipped {skipped} entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
