"""Semantic filters for post-hoc link verification (no training required).

Three approaches to filter false positives from word-sense ambiguity:

1. EmbeddingFilter: sentence-transformers cosine similarity between
   sentence and component's architectural context
2. TfidfFilter: TF-IDF vectors built from component context windows,
   compared with candidate sentence
3. LexicalFilter: Capitalization, standalone occurrence, and syntactic
   pattern heuristics
"""

import re
from collections import defaultdict


class LexicalFilter:
    """Rule-based word-sense disambiguation using lexical cues.

    Checks if a component name is used as an architectural reference vs
    a generic English word based on:
    - Capitalization: "Logic component" vs "cascade logic"
    - Standalone: "the Logic handles" vs "business logic"
    - Preceding words: "the X component" patterns
    - Following words: "X handles/processes/manages" patterns
    """

    # Words that signal architectural usage when preceding component name
    ARCH_PREFIXES = {
        "the", "a", "an", "our", "this", "each",
    }

    # Words that signal architectural usage when following component name
    ARCH_SUFFIXES = {
        "component", "module", "service", "server", "client", "layer",
        "subsystem", "package", "handles", "processes", "manages",
        "provides", "receives", "sends", "stores", "retrieves",
        "communicates", "connects", "serves",
    }

    # Words that signal generic usage when preceding component name
    GENERIC_PREFIXES = {
        "cascade", "business", "application", "core", "main",
        "internal", "external", "basic", "minimal", "simple",
        "complex", "general", "common", "shared", "global",
        "access", "data", "file", "user", "system",
    }

    def __init__(self, components, sentences, name_to_id):
        self.components = components
        self.name_to_id = name_to_id
        self.comp_names = {c.name for c in components}
        # Identify which names are also common English words (single short words)
        self._ambiguous_names = set()
        for c in components:
            name_lower = c.name.lower()
            # Single word, all lowercase when lowered, and short
            if ' ' not in c.name and '-' not in c.name and len(c.name) <= 10:
                if name_lower in _COMMON_WORDS:
                    self._ambiguous_names.add(c.name)

    def is_architectural_reference(self, sentence_text: str, comp_name: str) -> bool:
        """Check if comp_name in sentence is an architectural reference."""
        if comp_name not in self._ambiguous_names:
            return True  # Non-ambiguous names are always architectural

        text = sentence_text
        name_lower = comp_name.lower()

        # Check 1: Does name appear with original capitalization?
        if re.search(rf'\b{re.escape(comp_name)}\b', text):
            # Capitalized occurrence — likely architectural
            # But check if preceded by a generic modifier
            match = re.search(rf'(\w+)\s+{re.escape(comp_name)}\b', text)
            if match:
                preceding = match.group(1).lower()
                if preceding in self.GENERIC_PREFIXES:
                    return False
            return True

        # Check 2: Only lowercase occurrence
        if re.search(rf'\b{re.escape(name_lower)}\b', text.lower()):
            # Check if followed by architectural suffix
            match = re.search(rf'\b{re.escape(name_lower)}\s+(\w+)', text.lower())
            if match:
                following = match.group(1)
                if following in self.ARCH_SUFFIXES:
                    return True
            return False  # Lowercase without arch context = generic

        return True  # Name not found at all — shouldn't happen

    def filter_links(self, links):
        """Filter links, removing those with generic word-sense usage."""
        result = []
        for link in links:
            if self.is_architectural_reference(
                getattr(link, '_sentence_text', ''),
                link.component_name
            ):
                result.append(link)
        return result


class TfidfFilter:
    """TF-IDF based component context similarity.

    Builds a TF-IDF profile for each component from sentences that
    explicitly mention it (gold-standard-free — uses the document itself).
    Then checks if candidate sentences are contextually similar.
    """

    def __init__(self, components, sentences, name_to_id):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        self.cosine_similarity = cosine_similarity
        self.np = np
        self.comp_names = {c.name for c in components}
        self.name_to_id = name_to_id

        # Build component profiles from sentences that explicitly mention them
        comp_texts = defaultdict(list)
        for sent in sentences:
            text_lower = sent.text.lower()
            for c in components:
                if c.name.lower() in text_lower:
                    if re.search(rf'\b{re.escape(c.name)}\b', sent.text, re.IGNORECASE):
                        comp_texts[c.name].append(sent.text)

        # Build TF-IDF vectors
        all_docs = []
        self._comp_doc_indices = {}
        for cname, texts in comp_texts.items():
            self._comp_doc_indices[cname] = len(all_docs)
            all_docs.append(" ".join(texts))

        if not all_docs:
            self._vectorizer = None
            return

        self._vectorizer = TfidfVectorizer(
            max_features=500, stop_words='english',
            ngram_range=(1, 2), min_df=1
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(all_docs)

    def similarity(self, sentence_text: str, comp_name: str) -> float:
        """Compute TF-IDF similarity between sentence and component profile."""
        if self._vectorizer is None or comp_name not in self._comp_doc_indices:
            return 0.5  # Unknown — don't filter

        sent_vec = self._vectorizer.transform([sentence_text])
        comp_idx = self._comp_doc_indices[comp_name]
        comp_vec = self._tfidf_matrix[comp_idx]
        sim = self.cosine_similarity(sent_vec, comp_vec)[0, 0]
        return float(sim)


class EmbeddingFilter:
    """Sentence-transformer embedding similarity filter.

    Uses a pre-trained sentence transformer to compare sentence embeddings
    with component description embeddings. Component descriptions are built
    from sentences that explicitly mention the component.
    """

    def __init__(self, components, sentences, name_to_id, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        import numpy as np

        self.np = np
        self.name_to_id = name_to_id

        print(f"    Loading embedding model: {model_name}")
        self._model = SentenceTransformer(model_name)

        # Build component profiles from explicit mentions
        comp_texts = defaultdict(list)
        for sent in sentences:
            text_lower = sent.text.lower()
            for c in components:
                if c.name.lower() in text_lower:
                    if re.search(rf'\b{re.escape(c.name)}\b', sent.text, re.IGNORECASE):
                        comp_texts[c.name].append(sent.text)

        # Compute component embeddings (average of mention sentences)
        self._comp_embeddings = {}
        for cname, texts in comp_texts.items():
            if texts:
                embeddings = self._model.encode(texts)
                self._comp_embeddings[cname] = np.mean(embeddings, axis=0)

        # Pre-compute sentence embeddings
        self._sent_embeddings = {}
        all_texts = [s.text for s in sentences]
        if all_texts:
            all_embs = self._model.encode(all_texts)
            for sent, emb in zip(sentences, all_embs):
                self._sent_embeddings[sent.number] = emb

    def similarity(self, sentence_number: int, comp_name: str) -> float:
        """Compute embedding similarity between sentence and component profile."""
        if comp_name not in self._comp_embeddings:
            return 0.5  # Unknown component — don't filter

        sent_emb = self._sent_embeddings.get(sentence_number)
        if sent_emb is None:
            return 0.5

        comp_emb = self._comp_embeddings[comp_name]
        # Cosine similarity
        sim = float(self.np.dot(sent_emb, comp_emb) /
                     (self.np.linalg.norm(sent_emb) * self.np.linalg.norm(comp_emb) + 1e-8))
        return sim


# Common English words that might also be component names
_COMMON_WORDS = {
    "common", "logic", "storage", "client", "server", "test", "driver",
    "core", "web", "app", "apps", "ui", "db", "api", "model", "view",
    "controller", "service", "registry", "gateway", "proxy", "cache",
    "queue", "bus", "hub", "bridge", "adapter", "facade", "factory",
    "builder", "observer", "listener", "handler", "manager", "provider",
    "consumer", "producer", "worker", "scheduler", "monitor", "logger",
    "parser", "validator", "converter", "mapper", "router", "filter",
    "interceptor", "middleware", "plugin", "extension", "module",
    "component", "entity", "repository", "store", "persistence",
    "preferences", "recommender",
}
