"""Document and TransArc loading utilities."""

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .data_types import SadSamLink


@dataclass
class Sentence:
    """A sentence from documentation."""
    number: int  # 1-indexed
    text: str

    def has_pronoun(self) -> bool:
        """Check if sentence contains pronouns that might need resolution."""
        pattern = r'\b(it|they|this|these|that|those|its|their)\b'
        return bool(re.search(pattern, self.text.lower()))

    def get_words(self) -> list[str]:
        """Get list of words in sentence."""
        return re.findall(r'\b\w+\b', self.text)


class DocumentLoader:
    """Load and parse documentation and TransArc results."""

    @staticmethod
    def load_sentences(doc_path: str) -> list[Sentence]:
        """Load sentences from documentation file.

        Args:
            doc_path: Path to documentation text file (one sentence per line)

        Returns:
            List of Sentence objects
        """
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

    @staticmethod
    def load_transarc(
        transarc_csv: str,
        id_to_name: dict[str, str],
        name_to_id: dict[str, str],
        sent_map: dict[int, Sentence],
        model_knowledge: Optional[object] = None
    ) -> list[SadSamLink]:
        """Load TransArc baseline links.

        Args:
            transarc_csv: Path to TransArc CSV file
            id_to_name: Map from component ID to name
            name_to_id: Map from component name to ID
            sent_map: Map from sentence number to Sentence
            model_knowledge: Optional ModelKnowledge for impl->abstract mapping

        Returns:
            List of SadSamLink objects
        """
        links = []

        if not transarc_csv or not os.path.exists(transarc_csv):
            return links

        with open(transarc_csv) as f:
            for row in csv.DictReader(f):
                cid = row.get('modelElementID', '')
                snum = int(row.get('sentence', 0))
                cname = id_to_name.get(cid, '')

                if not cname or snum not in sent_map:
                    continue

                # Skip if component appears in dotted path (package reference)
                sent = sent_map[snum]
                if DocumentLoader._in_dotted_path(sent.text, cname):
                    continue

                # Map implementation to abstract if available
                if model_knowledge and hasattr(model_knowledge, 'impl_to_abstract'):
                    if cname in model_knowledge.impl_to_abstract:
                        cname = model_knowledge.impl_to_abstract[cname]
                        cid = name_to_id.get(cname, cid)

                # Confidence based on architectural status
                conf = 0.92
                if model_knowledge and hasattr(model_knowledge, 'architectural_names'):
                    if cname in model_knowledge.architectural_names:
                        conf = 0.94

                links.append(SadSamLink(snum, cid, cname, conf, "transarc"))

        return links

    @staticmethod
    def _in_dotted_path(text: str, comp_name: str) -> bool:
        """Check if component name appears in a dotted path (e.g., package.class)."""
        for path in re.findall(r'\b\w+(?:\.\w+)+\b', text.lower()):
            if comp_name.lower() in path.split('.'):
                return True
        return False

    @staticmethod
    def detect_paragraphs(sentences: list[Sentence]) -> list[list[Sentence]]:
        """Group sentences into paragraphs based on content shifts.

        Simple heuristic: new paragraph when sentence is significantly shorter
        or starts with a transitional phrase.

        Args:
            sentences: List of sentences

        Returns:
            List of paragraph (list of sentences)
        """
        if not sentences:
            return []

        paragraphs = []
        current_para = [sentences[0]]

        transition_phrases = [
            'however', 'furthermore', 'additionally', 'in addition',
            'moreover', 'on the other hand', 'in contrast', 'similarly',
            'the following', 'as mentioned', 'as described'
        ]

        for i in range(1, len(sentences)):
            sent = sentences[i]
            prev = sentences[i - 1]

            # Check for paragraph boundary indicators
            is_new_para = False

            # Short sentence after long one might be new topic
            if len(prev.text) > 100 and len(sent.text) < 50:
                is_new_para = True

            # Transitional phrase at start
            sent_lower = sent.text.lower()
            if any(sent_lower.startswith(phrase) for phrase in transition_phrases):
                is_new_para = True

            # Numbered list item
            if re.match(r'^\d+\.?\s', sent.text):
                is_new_para = True

            if is_new_para and current_para:
                paragraphs.append(current_para)
                current_para = []

            current_para.append(sent)

        if current_para:
            paragraphs.append(current_para)

        return paragraphs

    @staticmethod
    def build_sent_map(sentences: list[Sentence]) -> dict[int, Sentence]:
        """Build sentence number to Sentence mapping."""
        return {s.number: s for s in sentences}

    @staticmethod
    def get_context_window(
        sentences: list[Sentence],
        current_idx: int,
        window_before: int = 2,
        window_after: int = 1
    ) -> list[Sentence]:
        """Get sentences in context window around current sentence.

        Args:
            sentences: All sentences
            current_idx: Index of current sentence (0-based)
            window_before: Number of sentences before
            window_after: Number of sentences after

        Returns:
            List of sentences in window
        """
        start = max(0, current_idx - window_before)
        end = min(len(sentences), current_idx + window_after + 1)
        return sentences[start:end]
