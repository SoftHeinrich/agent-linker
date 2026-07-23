"""Document loading utilities for the S-Linker family (v2).

Clean-slate replacement for core/document_loader.py — no dead methods,
no dead Sentence helpers.  Shared foundation for S-Linker11+ linkers.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sentence:
    """A sentence from documentation."""
    number: int  # 1-indexed
    text: str


def load_sentences(doc_path: str) -> list[Sentence]:
    """Load sentences from documentation file (one sentence per line)."""
    path = Path(doc_path)
    if not path.exists():
        raise FileNotFoundError(f"Documentation file not found: {doc_path}")

    sentences = []
    with open(path, encoding='utf-8') as f:
        sent_num = 0
        for line in f:
            text = line.strip()
            if text:
                sent_num += 1
                sentences.append(Sentence(number=sent_num, text=text))

    return sentences


def build_sent_map(sentences: list[Sentence]) -> dict[int, Sentence]:
    """Build sentence number to Sentence mapping."""
    return {s.number: s for s in sentences}
