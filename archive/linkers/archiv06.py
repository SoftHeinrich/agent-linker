"""AgentLinker V2 Ablation: V2 (qualitative) with adaptive coref/implicit modes.

Extends AgentLinkerV2 with mode switching for ablation studies:
- coref_mode: "discourse" (default), "debate", "adaptive"
- implicit_mode: "on" (default), "off", "adaptive"
- recovery_mode: "on", "off", "skip_ambiguous" (default V2 behavior)
- post_filter: "none" (default), "embedding", "tfidf", "lexical"

Adaptive modes use sentences-per-component ratio to decide behavior.
"""

from typing import Optional

from ...core import SadSamLink, Sentence
from ...llm_client import LLMBackend
from .agent_linker_v2 import AgentLinkerV2
from .agent_linker_ablation import AgentLinkerAblation


class AgentLinkerV2Ablation(AgentLinkerV2):
    """V2 with adaptive coref/implicit mode switching and semantic filters."""

    def __init__(
        self,
        backend: Optional[LLMBackend] = None,
        coref_mode: str = None,
        implicit_mode: str = None,
        recovery_mode: str = None,
        post_filter: str = None,
        **kwargs,
    ):
        super().__init__(backend=backend)
        self.coref_mode = coref_mode or "discourse"
        self.implicit_mode = implicit_mode or "on"
        self.recovery_mode = recovery_mode or "skip_ambiguous"
        self.post_filter = post_filter or "none"
        self._semantic_filter = None
        print(f"  V2 Ablation: coref={self.coref_mode}, implicit={self.implicit_mode}, "
              f"recovery={self.recovery_mode}, filter={self.post_filter}")

    def _is_complex_doc(self) -> bool:
        if not self.doc_profile:
            return False
        spc = self.doc_profile.sentence_count / max(1, self.doc_profile.component_count)
        return spc > 10

    def _init_semantic_filter(self, components, sentences, name_to_id):
        """Initialize the semantic filter after data is loaded."""
        if self.post_filter == "none":
            return
        if self.post_filter == "embedding":
            from .semantic_filters import EmbeddingFilter
            self._semantic_filter = EmbeddingFilter(components, sentences, name_to_id)
        elif self.post_filter == "tfidf":
            from .semantic_filters import TfidfFilter
            self._semantic_filter = TfidfFilter(components, sentences, name_to_id)
        elif self.post_filter == "lexical":
            from .semantic_filters import LexicalFilter
            self._semantic_filter = LexicalFilter(components, sentences, name_to_id)

    def link(self, text_path, model_path, transarc_csv=None):
        """Override link to initialize semantic filter and apply post-filtering."""
        from ...pcm_parser import parse_pcm_repository
        from ...core import DocumentLoader

        # Pre-load data to init filter before pipeline runs
        components = parse_pcm_repository(model_path)
        sentences = DocumentLoader.load_sentences(text_path)
        name_to_id = {c.name: c.id for c in components}
        sent_map = {s.number: s for s in sentences}

        if self.post_filter != "none":
            print(f"\n[Semantic Filter Init] {self.post_filter}")
            self._init_semantic_filter(components, sentences, name_to_id)

        # Run full pipeline
        result = super().link(text_path, model_path, transarc_csv)

        # Apply post-filter
        if self._semantic_filter is not None and result:
            before = len(result)
            result = self._apply_semantic_filter(result, sent_map)
            print(f"  Semantic filter ({self.post_filter}): {before} -> {len(result)} "
                  f"({before - len(result)} removed)")

        return result

    def _apply_semantic_filter(self, links, sent_map):
        """Apply semantic filter to remove FPs from word-sense ambiguity."""
        from .semantic_filters import EmbeddingFilter, TfidfFilter, LexicalFilter

        ambiguous = self.model_knowledge.ambiguous_names if self.model_knowledge else set()
        # Also check which names are common English words
        from .semantic_filters import _COMMON_WORDS
        short_names = {c.component_name for c in links
                       if c.component_name.lower() in _COMMON_WORDS}
        names_to_check = ambiguous | short_names

        if not names_to_check:
            return links  # Nothing to filter

        result = []
        for link in links:
            if link.component_name not in names_to_check:
                result.append(link)
                continue

            # Apply filter based on type
            keep = True
            if isinstance(self._semantic_filter, EmbeddingFilter):
                sim = self._semantic_filter.similarity(link.sentence_number, link.component_name)
                keep = sim >= 0.3  # Conservative threshold
                if not keep:
                    print(f"    Embedding filter rejected: S{link.sentence_number} -> "
                          f"{link.component_name} (sim={sim:.3f})")
            elif isinstance(self._semantic_filter, TfidfFilter):
                sent = sent_map.get(link.sentence_number)
                if sent:
                    sim = self._semantic_filter.similarity(sent.text, link.component_name)
                    keep = sim >= 0.1  # TF-IDF scores are lower
                    if not keep:
                        print(f"    TF-IDF filter rejected: S{link.sentence_number} -> "
                              f"{link.component_name} (sim={sim:.3f})")
            elif isinstance(self._semantic_filter, LexicalFilter):
                sent = sent_map.get(link.sentence_number)
                if sent:
                    keep = self._semantic_filter.is_architectural_reference(
                        sent.text, link.component_name
                    )
                    if not keep:
                        print(f"    Lexical filter rejected: S{link.sentence_number} -> "
                              f"{link.component_name}")

            if keep:
                result.append(link)

        return result

    def _resolve_coreferences_with_discourse(
        self, sentences: list[Sentence], components: list, name_to_id: dict,
        sent_map: dict, discourse_model: dict
    ) -> list[SadSamLink]:
        mode = self.coref_mode

        if mode == "adaptive":
            if self._is_complex_doc():
                mode = "debate"
                print(f"    Adaptive coref: debate (complex doc, {len(sentences)} sents)")
            else:
                mode = "discourse"
                print(f"    Adaptive coref: discourse ({len(sentences)} sents)")

        if mode == "debate":
            return AgentLinkerAblation._resolve_coreferences_with_debate(
                self, sentences, components, name_to_id, sent_map
            )
        else:
            return super()._resolve_coreferences_with_discourse(
                sentences, components, name_to_id, sent_map, discourse_model
            )

    def _detect_implicit_references(
        self, sentences, components, name_to_id, sent_map, discourse_model, existing_links
    ) -> list[SadSamLink]:
        skip = False
        if self.implicit_mode == "off":
            skip = True
        elif self.implicit_mode == "adaptive":
            if self._is_complex_doc():
                skip = True
                print(f"    SKIPPED (adaptive: complex doc, {len(sentences)} sentences)")
            else:
                print(f"    ENABLED (adaptive: {len(sentences)} sentences)")

        if skip:
            if self.implicit_mode == "off":
                print("    SKIPPED (ablation: no_implicit)")
            return []
        return super()._detect_implicit_references(
            sentences, components, name_to_id, sent_map, discourse_model, existing_links
        )

    def _validate_implicit_links(
        self, implicit_links, sentences, components, sent_map
    ) -> list[SadSamLink]:
        if self.implicit_mode == "off":
            return []
        if self.implicit_mode == "adaptive" and self._is_complex_doc():
            return []
        return super()._validate_implicit_links(implicit_links, sentences, components, sent_map)

    def _adaptive_fn_recovery(self, current_links, sentences, components, name_to_id, sent_map) -> list[SadSamLink]:
        if self.recovery_mode == "off":
            print("    SKIPPED (ablation: recovery=off)")
            return current_links
        if self.recovery_mode == "on":
            from .agent_linker import AgentLinker
            return AgentLinker._adaptive_fn_recovery(self, current_links, sentences, components, name_to_id, sent_map)
        return super()._adaptive_fn_recovery(current_links, sentences, components, name_to_id, sent_map)
