#!/usr/bin/env python3
"""Require active paper table and figure captions to be one sentence under 15 words."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
MAX_WORDS = 14


def captions(text: str):
    """Yield complete ``\\caption{...}`` contents, including nested TeX braces."""
    start = 0
    marker = "\\caption{"
    while (found := text.find(marker, start)) != -1:
        i = found + len(marker)
        depth = 1
        while i < len(text) and depth:
            depth += (text[i] == "{") - (text[i] == "}")
            i += 1
        if depth:
            raise ValueError("unclosed \\caption")
        yield text[found + len(marker):i - 1]
        start = i


def plain_text(caption: str) -> str:
    caption = re.sub(r"\\[A-Za-z]+(?:\[[^]]*\])?(?:\{[^{}]*\})?", "TERM", caption)
    return re.sub(r"\s+", " ", caption).strip()


def main() -> None:
    failures = []
    checked = 0
    for path in sorted(PAPER.rglob("*.tex")):
        if "archive" in path.parts:
            continue
        for caption in captions(path.read_text(encoding="utf-8")):
            checked += 1
            text = plain_text(caption)
            words = re.findall(r"[A-Za-z0-9]+", text)
            sentences = re.findall(r"[.!?](?=\s|$)", text)
            if len(words) > MAX_WORDS or len(sentences) != 1:
                failures.append(
                    f"{path.relative_to(ROOT)}: {len(words)} words, "
                    f"{len(sentences)} sentence markers: {text}"
                )
    if failures:
        raise SystemExit("Caption check failed:\n" + "\n".join(failures))
    print(f"PASS: {checked} active captions are one sentence with at most {MAX_WORDS} words.")


if __name__ == "__main__":
    main()
