"""ILinker2 V30d: V30c + resume support + CamelCase Phase 3 override.

Reads V30c pickle checkpoints to skip expensive earlier phases.
Adds back the CamelCase rescue override that V30c removed — CamelCase
identifiers (e.g., DataStorage, AudioAccess) are constructed proper names
and should never be rejected as generic by the LLM judge.

Usage:
  # Full run (same as V30c but saves to v30d/ checkpoints)
  linker.link(text_path, model_path)

  # Resume from Phase 3 using V30c checkpoints (re-runs Phase 3+ with CamelCase fix)
  linker.link(text_path, model_path, resume_from_phase=3)

  # Resume from pre-judge (re-runs Phase 9+ only)
  linker.link(text_path, model_path, resume_from_phase=9)
"""

import os
import pickle
import re
import time
from collections import defaultdict

from llm_sad_sam.core.data_types import (
    SadSamLink, CandidateLink, DocumentKnowledge, LearnedThresholds,
)
from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.linkers.experimental.agent_linker_v26a import AgentLinkerV26a
from llm_sad_sam.linkers.experimental.ilinker2 import ILinker2
from llm_sad_sam.pcm_parser import parse_pcm_repository


# Phase name -> numeric index for resume comparison
PHASE_INDEX = {
    "phase0": 0, "phase1": 1, "phase2": 2, "phase3": 3, "phase4": 4,
    "phase5": 5, "phase6": 6, "phase7": 7, "pre_judge": 9, "final": 10,
}
# Numeric index -> checkpoint name to load before that phase
PHASE_CHECKPOINT = {
    0: None,        # nothing to load
    1: "phase0",    # load phase0 checkpoint, run from phase 1
    2: "phase1",
    3: "phase2",    # load phases 0-2, re-run phase 3+
    4: "phase3",
    5: "phase4",
    6: "phase5",
    7: "phase6",
    8: "phase7",    # load through phase 7, run from phase 8
    9: "pre_judge", # load pre-judge, run phase 9+
}


class ILinker2V30d(AgentLinkerV26a):
    """V30c + resume from checkpoint + CamelCase Phase 3 override."""

    def __init__(self, checkpoint_source="v30c",
                 enable_camelcase=True,      # Fix A: CamelCase override
                 enable_v24=False,           # V24: exact-component + multi-word proper-name
                 enable_uppercase=False,     # Fix B: uppercase ≤4 chars override
                 partial_min_count=3,        # Phase 3b: multiword partial threshold
                 # Heuristic toggles for ablation testing
                 disable_generic_mention=False,    # Phase 6: _is_generic_mention
                 disable_generic_coref=False,      # Phase 7: _filter_generic_coref
                 disable_pronoun_coref=False,       # Phase 7: _deterministic_pronoun_coref
                 disable_boundary_filters=False,    # Phase 8c: all boundary filters
                 disable_partial_injection=False,   # Phase 8b: _inject_partial_references
                 disable_syn_safe=False,            # Phase 9: synonym-safe judge bypass
                 disable_parent_overlap=False,      # Between 8b-8c: parent-overlap guard
                 disable_abbrev_guard=False,         # Phase 5: abbreviation guard
                 **kwargs):
        super().__init__(**kwargs)
        self._ilinker2 = ILinker2(backend=self.llm.backend)
        self._checkpoint_source = checkpoint_source
        self._enable_camelcase = enable_camelcase
        self._enable_v24 = enable_v24
        self._enable_uppercase = enable_uppercase
        self._partial_min_count = partial_min_count
        # Heuristic toggles
        self._disable_generic_mention = disable_generic_mention
        self._disable_generic_coref = disable_generic_coref
        self._disable_pronoun_coref = disable_pronoun_coref
        self._disable_boundary_filters = disable_boundary_filters
        self._disable_partial_injection = disable_partial_injection
        self._disable_syn_safe = disable_syn_safe
        self._disable_parent_overlap = disable_parent_overlap
        self._disable_abbrev_guard = disable_abbrev_guard

    # ── Checkpoint helpers ───────────────────────────────────────────────

    def _checkpoint_dir(self, text_path, variant=None):
        cache_dir = os.environ.get("PHASE_CACHE_DIR", "./results/phase_cache")
        ds = os.path.splitext(os.path.basename(text_path))[0]
        v = variant or "v30d"
        d = os.path.join(cache_dir, v, ds)
        os.makedirs(d, exist_ok=True)
        return d

    def _save_phase(self, text_path, phase_name, state):
        d = self._checkpoint_dir(text_path)
        path = os.path.join(d, f"{phase_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"  Checkpoint: {phase_name} saved")

    def _load_phase(self, text_path, phase_name, variant=None):
        v = variant or "v30d"
        d = self._checkpoint_dir(text_path, variant=v)
        path = os.path.join(d, f"{phase_name}.pkl")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def _restore_state(self, text_path, up_to_phase, components):
        """Load all checkpoints up to (but not including) the given phase number.

        Tries v30d checkpoints first, falls back to checkpoint_source (v30c).
        """
        # Map: which checkpoints we need for each phase
        needed = []
        if up_to_phase > 0:
            needed.append("phase0")
        if up_to_phase > 1:
            needed.append("phase1")
        if up_to_phase > 2:
            needed.append("phase2")
        if up_to_phase > 3:
            needed.append("phase3")
        if up_to_phase > 4:
            needed.append("phase4")
        if up_to_phase > 5:
            needed.append("phase5")
        if up_to_phase > 6:
            needed.append("phase6")
        if up_to_phase > 7:
            needed.append("phase7")
        if up_to_phase > 8:
            needed.append("pre_judge")

        for ckpt_name in needed:
            # Try own checkpoints first, then source
            data = self._load_phase(text_path, ckpt_name)
            if data is None:
                data = self._load_phase(text_path, ckpt_name, variant=self._checkpoint_source)
            if data is None:
                print(f"  WARNING: checkpoint {ckpt_name} not found in v30d or {self._checkpoint_source}")
                return False
            self._apply_checkpoint(data, components)
            print(f"  Restored: {ckpt_name}")

        return True

    def _apply_checkpoint(self, data, components):
        """Apply a single checkpoint dict to instance state."""
        if "doc_profile" in data:
            self.doc_profile = data["doc_profile"]
        if "is_complex" in data:
            self._is_complex = data["is_complex"]
        if "model_knowledge" in data:
            self.model_knowledge = data["model_knowledge"]
        if "generic_component_words" in data:
            self.GENERIC_COMPONENT_WORDS = data["generic_component_words"]
        if "generic_partials" in data:
            self.GENERIC_PARTIALS = data["generic_partials"]
        if "learned_patterns" in data:
            self.learned_patterns = data["learned_patterns"]
        if "doc_knowledge" in data:
            self.doc_knowledge = data["doc_knowledge"]
        # Link data stored as local vars, returned via the checkpoint
        # These are handled by the caller via the data dict

    # ── NDF: disable dotted-path regex ───────────────────────────────────

    def _in_dotted_path(self, text: str, comp_name: str) -> bool:
        return False

    # ── Heuristic overrides for ablation ──────────────────────────────────

    def _is_generic_mention(self, comp_name, sentence_text):
        if self._disable_generic_mention:
            return False
        return super()._is_generic_mention(comp_name, sentence_text)

    def _filter_generic_coref(self, coref_links, sent_map):
        if self._disable_generic_coref:
            return coref_links
        return super()._filter_generic_coref(coref_links, sent_map)

    def _deterministic_pronoun_coref(self, sentences, components, name_to_id, sent_map, existing_set):
        if self._disable_pronoun_coref:
            return []
        return super()._deterministic_pronoun_coref(sentences, components, name_to_id, sent_map, existing_set)

    def _apply_boundary_filters(self, preliminary, sent_map, transarc_set):
        if self._disable_boundary_filters:
            return preliminary, []
        return super()._apply_boundary_filters(preliminary, sent_map, transarc_set)

    def _inject_partial_references(self, sentences, components, name_to_id,
                                    transarc_set, entity_set, coref_set, implicit_set):
        if self._disable_partial_injection:
            return []
        return super()._inject_partial_references(
            sentences, components, name_to_id, transarc_set, entity_set, coref_set, implicit_set)

    def _apply_abbreviation_guard_to_candidates(self, candidates, sent_map):
        if self._disable_abbrev_guard:
            return candidates
        return super()._apply_abbreviation_guard_to_candidates(candidates, sent_map)

    def _has_alias_mention(self, comp_name, sentence_text):
        if self._disable_syn_safe:
            return False
        return super()._has_alias_mention(comp_name, sentence_text)

    # ── Single-phase execution ────────────────────────────────────────────

    def run_single_phase(self, text_path, model_path, phase):
        """Run ONLY the specified phase. Loads inputs from checkpoints, saves output, stops.

        Args:
            phase: int (3=Phase 3, 6=Phase 6, 9=Phase 9, etc.)

        Returns:
            The phase output dict (same as what gets pickled).
        """
        self._cached_text_path = text_path
        self._cached_model_path = model_path
        self._phase_log = []

        components = parse_pcm_repository(model_path)
        sentences = DocumentLoader.load_sentences(text_path)
        name_to_id = {c.name: c.id for c in components}
        id_to_name = {c.id: c.name for c in components}
        sent_map = DocumentLoader.build_sent_map(sentences)
        self._cached_sent_map = sent_map
        self.thresholds = LearnedThresholds(0, 0, 0, 0, "qualitative", 0)

        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        # Restore all state up to this phase
        ok = self._restore_state(text_path, phase, components)
        if not ok:
            raise RuntimeError(f"Cannot run phase {phase}: missing input checkpoints")
        print(f"\n  *** Single-phase mode: running ONLY phase {phase} ***\n")

        if phase == 0:
            print("[Phase 0] Document Profile")
            self.doc_profile = self._learn_document_profile(sentences, components)
            self._is_complex = self._structural_complexity(sentences, components)
            spc = len(sentences) / max(1, len(components))
            print(f"  Stats: {spc:.1f} sents/comp, {self.doc_profile.pronoun_ratio:.0%} pronouns")
            print(f"  Complex: {self._is_complex}")
            result = {"doc_profile": self.doc_profile, "is_complex": self._is_complex}
            self._save_phase(text_path, "phase0", result)
            return result

        elif phase == 1:
            print("[Phase 1] Model Structure")
            self.model_knowledge = self._analyze_model(components)
            arch = self.model_knowledge.architectural_names
            ambig = self.model_knowledge.ambiguous_names
            self.GENERIC_COMPONENT_WORDS = set()
            for name in ambig:
                if ' ' not in name and not name.isupper():
                    self.GENERIC_COMPONENT_WORDS.add(name.lower())
            self.GENERIC_PARTIALS = set()
            for comp in components:
                parts = re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z]|$)', comp.name)
                for part in parts:
                    p_lower = part.lower()
                    if part.isupper():
                        continue
                    if len(p_lower) >= 3 and p_lower in ambig or any(
                        p_lower == a.lower() for a in ambig
                    ):
                        self.GENERIC_PARTIALS.add(p_lower)
            for name in ambig:
                if ' ' not in name and not name.isupper():
                    self.GENERIC_PARTIALS.add(name.lower())
            print(f"  Architectural: {len(arch)}, Ambiguous: {sorted(ambig)}")
            print(f"  Generic words: {sorted(self.GENERIC_COMPONENT_WORDS)}")
            print(f"  Generic partials: {sorted(self.GENERIC_PARTIALS)}")
            result = {
                "model_knowledge": self.model_knowledge,
                "generic_component_words": self.GENERIC_COMPONENT_WORDS,
                "generic_partials": self.GENERIC_PARTIALS,
            }
            self._save_phase(text_path, "phase1", result)
            return result

        elif phase == 2:
            print("[Phase 2] Pattern Learning")
            self.learned_patterns = self._learn_patterns_with_debate(sentences, components)
            print(f"  Subprocess terms: {len(self.learned_patterns.subprocess_terms)}")
            result = {"learned_patterns": self.learned_patterns}
            self._save_phase(text_path, "phase2", result)
            return result

        elif phase == 3:
            print("[Phase 3] Document Knowledge")
            self.doc_knowledge = self._learn_document_knowledge_enriched(sentences, components)
            print(f"  Abbrev: {len(self.doc_knowledge.abbreviations)}, "
                  f"Syn: {len(self.doc_knowledge.synonyms)}, "
                  f"Generic: {len(self.doc_knowledge.generic_terms)}")
            self._enrich_multiword_partials(sentences, components)
            result = {"doc_knowledge": self.doc_knowledge}
            self._save_phase(text_path, "phase3", result)
            return result

        elif phase == 4:
            print("[Phase 4] ILinker2 Extraction")
            transarc_links = self._process_transarc(None, id_to_name, sent_map, name_to_id)
            transarc_set = {(l.sentence_number, l.component_id) for l in transarc_links}
            print(f"  Links: {len(transarc_links)}")
            result = {"transarc_links": transarc_links, "transarc_set": transarc_set}
            self._save_phase(text_path, "phase4", result)
            return result

        elif phase == 5:
            print("[Phase 5] Entity Extraction")
            # Need transarc_links for Phase 5b
            data4 = self._load_phase(text_path, "phase4") or self._load_phase(text_path, "phase4", variant=self._checkpoint_source)
            transarc_links = data4["transarc_links"]
            candidates = self._extract_entities_enriched(sentences, components, name_to_id, sent_map)
            print(f"  Candidates: {len(candidates)}")
            before_guard = len(candidates)
            candidates = self._apply_abbreviation_guard_to_candidates(candidates, sent_map)
            if len(candidates) < before_guard:
                print(f"  After abbrev guard: {len(candidates)} (-{before_guard - len(candidates)})")
            entity_comps = {c.component_name for c in candidates}
            transarc_comps = {l.component_name for l in transarc_links}
            unlinked = [c for c in components if c.name not in entity_comps | transarc_comps]
            if unlinked:
                print(f"\n[Phase 5b] Targeted Recovery ({len(unlinked)} unlinked)")
                extra = self._targeted_extraction(unlinked, sentences, name_to_id, sent_map,
                                                  components=components, transarc_links=transarc_links,
                                                  entity_candidates=candidates)
                if extra:
                    print(f"  Found: {len(extra)} additional")
                    candidates.extend(extra)
            result = {"candidates": candidates}
            self._save_phase(text_path, "phase5", result)
            return result

        elif phase == 6:
            print("[Phase 6] Validation")
            data5 = self._load_phase(text_path, "phase5") or self._load_phase(text_path, "phase5", variant=self._checkpoint_source)
            candidates = data5["candidates"]
            validated = self._validate_intersect(candidates, components, sent_map)
            print(f"  Validated: {len(validated)} (of {len(candidates)})")
            result = {"validated": validated}
            self._save_phase(text_path, "phase6", result)
            return result

        elif phase == 7:
            print("[Phase 7] Coreference")
            data4 = self._load_phase(text_path, "phase4") or self._load_phase(text_path, "phase4", variant=self._checkpoint_source)
            data6 = self._load_phase(text_path, "phase6") or self._load_phase(text_path, "phase6", variant=self._checkpoint_source)
            transarc_set = data4["transarc_set"]
            validated = data6["validated"]

            if self._is_complex:
                print(f"  Mode: debate (complex, {len(sentences)} sents)")
                coref_links = self._coref_debate(sentences, components, name_to_id, sent_map)
            else:
                discourse_model = self._build_discourse_model(sentences, components, name_to_id)
                print(f"  Mode: discourse ({len(sentences)} sents)")
                coref_links = self._coref_discourse(sentences, components, name_to_id, sent_map, discourse_model)
            before_coref = len(coref_links)
            coref_links = self._filter_generic_coref(coref_links, sent_map)
            if len(coref_links) < before_coref:
                print(f"  After generic filter: {len(coref_links)} (-{before_coref - len(coref_links)})")
            coref_set = {(l.sentence_number, l.component_id) for l in coref_links}
            pronoun_links = self._deterministic_pronoun_coref(
                sentences, components, name_to_id, sent_map,
                transarc_set | {(c.sentence_number, c.component_id) for c in validated} | coref_set)
            if pronoun_links:
                coref_links.extend(pronoun_links)
                print(f"  Deterministic pronoun coref: +{len(pronoun_links)}")
            print(f"  Coref links: {len(coref_links)}")
            result = {"coref_links": coref_links}
            self._save_phase(text_path, "phase7", result)
            return result

        elif phase == 9:
            print("[Phase 9] Judge Review")
            data = self._load_phase(text_path, "pre_judge") or self._load_phase(text_path, "pre_judge", variant=self._checkpoint_source)
            preliminary = data["preliminary"]
            transarc_set = data["transarc_set"]
            reviewed = self._judge_review(preliminary, sentences, components, sent_map, transarc_set)
            rejected = [l for l in preliminary if (l.sentence_number, l.component_id)
                        not in {(r.sentence_number, r.component_id) for r in reviewed}]
            print(f"  Approved: {len(reviewed)} (rejected {len(rejected)})")
            result = {"final": reviewed, "reviewed": reviewed, "rejected": rejected}
            self._save_phase(text_path, "final", result)
            return result

        else:
            raise ValueError(f"Single-phase mode not implemented for phase {phase}")

    # ── Main pipeline with resume support ─────────────────────────────────

    def link(self, text_path, model_path, transarc_csv=None, resume_from_phase=None):
        self._cached_text_path = text_path
        self._cached_model_path = model_path
        self._phase_log = []
        t0 = time.time()

        components = parse_pcm_repository(model_path)
        sentences = DocumentLoader.load_sentences(text_path)
        name_to_id = {c.name: c.id for c in components}
        id_to_name = {c.id: c.name for c in components}
        sent_map = DocumentLoader.build_sent_map(sentences)
        self._cached_sent_map = sent_map

        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        rfp = resume_from_phase or 0
        if rfp > 0:
            ok = self._restore_state(text_path, rfp, components)
            if not ok:
                raise RuntimeError(f"Cannot resume from phase {rfp}: missing checkpoints")
            self.thresholds = LearnedThresholds(0, 0, 0, 0, "qualitative", 0)
            print(f"\n  *** Resumed — will re-run from phase {rfp} ***\n")

        # Local vars that may come from checkpoints
        transarc_links = None
        transarc_set = None
        candidates = None
        validated = None
        coref_links = None
        preliminary = None

        # ── Phase 0 ─────────────────────────────────────────────────────
        if rfp <= 0:
            print("\n[Phase 0] Document Profile")
            self.doc_profile = self._learn_document_profile(sentences, components)
            self._is_complex = self._structural_complexity(sentences, components)
            spc = len(sentences) / max(1, len(components))
            print(f"  Stats: {spc:.1f} sents/comp, {self.doc_profile.pronoun_ratio:.0%} pronouns")
            print(f"  Complex: {self._is_complex}")
            self.thresholds = LearnedThresholds(0, 0, 0, 0, "qualitative", 0)
            self._save_phase(text_path, "phase0", {
                "doc_profile": self.doc_profile,
                "is_complex": self._is_complex,
            })
        else:
            print(f"\n[Phase 0] — skipped (resumed)")

        # ── Phase 1 ─────────────────────────────────────────────────────
        if rfp <= 1:
            print("\n[Phase 1] Model Structure")
            self.model_knowledge = self._analyze_model(components)
            arch = self.model_knowledge.architectural_names
            ambig = self.model_knowledge.ambiguous_names
            self.GENERIC_COMPONENT_WORDS = set()
            for name in ambig:
                if ' ' not in name and not name.isupper():
                    self.GENERIC_COMPONENT_WORDS.add(name.lower())
            print(f"  Architectural: {len(arch)}, Ambiguous: {sorted(ambig)}")
            print(f"  Discovered generic component words: {sorted(self.GENERIC_COMPONENT_WORDS)}")
            self.GENERIC_PARTIALS = set()
            for comp in components:
                parts = re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z]|$)', comp.name)
                for part in parts:
                    p_lower = part.lower()
                    if part.isupper():
                        continue
                    if len(p_lower) >= 3 and p_lower in ambig or any(
                        p_lower == a.lower() for a in ambig
                    ):
                        self.GENERIC_PARTIALS.add(p_lower)
            for name in ambig:
                if ' ' not in name and not name.isupper():
                    self.GENERIC_PARTIALS.add(name.lower())
            print(f"  Discovered generic partials: {sorted(self.GENERIC_PARTIALS)}")
            self._save_phase(text_path, "phase1", {
                "model_knowledge": self.model_knowledge,
                "generic_component_words": self.GENERIC_COMPONENT_WORDS,
                "generic_partials": self.GENERIC_PARTIALS,
            })
        else:
            print(f"\n[Phase 1] — skipped (resumed)")

        # ── Phase 2 ─────────────────────────────────────────────────────
        if rfp <= 2:
            print("\n[Phase 2] Pattern Learning")
            self.learned_patterns = self._learn_patterns_with_debate(sentences, components)
            print(f"  Subprocess terms: {len(self.learned_patterns.subprocess_terms)}")
            self._save_phase(text_path, "phase2", {
                "learned_patterns": self.learned_patterns,
            })
        else:
            print(f"\n[Phase 2] — skipped (resumed)")

        # ── Phase 3 ─────────────────────────────────────────────────────
        if rfp <= 3:
            print("\n[Phase 3] Document Knowledge")
            self.doc_knowledge = self._learn_document_knowledge_enriched(sentences, components)
            print(f"  Abbrev: {len(self.doc_knowledge.abbreviations)}, "
                  f"Syn: {len(self.doc_knowledge.synonyms)}, "
                  f"Generic: {len(self.doc_knowledge.generic_terms)}")
            self._enrich_multiword_partials(sentences, components)
            self._save_phase(text_path, "phase3", {
                "doc_knowledge": self.doc_knowledge,
            })
        else:
            print(f"\n[Phase 3] — skipped (resumed)")

        # ── Phase 4 ─────────────────────────────────────────────────────
        if rfp <= 4:
            print("\n[Phase 4] TransArc")
            transarc_links = self._process_transarc(transarc_csv, id_to_name, sent_map, name_to_id)
            transarc_set = {(l.sentence_number, l.component_id) for l in transarc_links}
            print(f"  Links: {len(transarc_links)}")
            self._save_phase(text_path, "phase4", {
                "transarc_links": transarc_links,
                "transarc_set": transarc_set,
            })
        else:
            data = self._load_phase(text_path, "phase4") or self._load_phase(text_path, "phase4", variant=self._checkpoint_source)
            transarc_links = data["transarc_links"]
            transarc_set = data["transarc_set"]
            print(f"\n[Phase 4] — skipped (resumed, {len(transarc_links)} links)")

        # ── Phase 5 ─────────────────────────────────────────────────────
        if rfp <= 5:
            print("\n[Phase 5] Entity Extraction")
            candidates = self._extract_entities_enriched(sentences, components, name_to_id, sent_map)
            print(f"  Candidates: {len(candidates)}")
            before_guard = len(candidates)
            candidates = self._apply_abbreviation_guard_to_candidates(candidates, sent_map)
            if len(candidates) < before_guard:
                print(f"  After abbrev guard: {len(candidates)} (-{before_guard - len(candidates)})")

            entity_comps = {c.component_name for c in candidates}
            transarc_comps = {l.component_name for l in transarc_links}
            covered_comps = entity_comps | transarc_comps
            unlinked = [c for c in components if c.name not in covered_comps]
            if unlinked:
                print(f"\n[Phase 5b] Targeted Recovery ({len(unlinked)} unlinked components)")
                extra = self._targeted_extraction(unlinked, sentences, name_to_id, sent_map,
                                                  components=components, transarc_links=transarc_links,
                                                  entity_candidates=candidates)
                if extra:
                    print(f"  Found: {len(extra)} additional candidates")
                    candidates.extend(extra)
                else:
                    print(f"  Found: 0 additional candidates")
            self._save_phase(text_path, "phase5", {"candidates": candidates})
        else:
            data = self._load_phase(text_path, "phase5") or self._load_phase(text_path, "phase5", variant=self._checkpoint_source)
            candidates = data["candidates"]
            print(f"\n[Phase 5] — skipped (resumed, {len(candidates)} candidates)")

        # ── Phase 6 ─────────────────────────────────────────────────────
        if rfp <= 6:
            print("\n[Phase 6] Validation")
            validated = self._validate_intersect(candidates, components, sent_map)
            print(f"  Validated: {len(validated)} (of {len(candidates)})")
            self._save_phase(text_path, "phase6", {"validated": validated})
        else:
            data = self._load_phase(text_path, "phase6") or self._load_phase(text_path, "phase6", variant=self._checkpoint_source)
            validated = data["validated"]
            print(f"\n[Phase 6] — skipped (resumed, {len(validated)} validated)")

        # ── Phase 7 ─────────────────────────────────────────────────────
        if rfp <= 7:
            print("\n[Phase 7] Coreference")
            if self._is_complex:
                print(f"  Mode: debate (complex, {len(sentences)} sents)")
                coref_links = self._coref_debate(sentences, components, name_to_id, sent_map)
            else:
                discourse_model = self._build_discourse_model(sentences, components, name_to_id)
                print(f"  Mode: discourse ({len(sentences)} sents)")
                coref_links = self._coref_discourse(sentences, components, name_to_id, sent_map, discourse_model)

            before_coref = len(coref_links)
            coref_links = self._filter_generic_coref(coref_links, sent_map)
            if len(coref_links) < before_coref:
                print(f"  After generic filter: {len(coref_links)} (-{before_coref - len(coref_links)})")

            coref_set = {(l.sentence_number, l.component_id) for l in coref_links}
            pronoun_links = self._deterministic_pronoun_coref(
                sentences, components, name_to_id, sent_map,
                transarc_set | {(c.sentence_number, c.component_id) for c in validated} | coref_set)
            if pronoun_links:
                coref_links.extend(pronoun_links)
                print(f"  Deterministic pronoun coref: +{len(pronoun_links)}")
            print(f"  Coref links: {len(coref_links)}")
            self._save_phase(text_path, "phase7", {"coref_links": coref_links})
        else:
            data = self._load_phase(text_path, "phase7") or self._load_phase(text_path, "phase7", variant=self._checkpoint_source)
            coref_links = data["coref_links"]
            print(f"\n[Phase 7] — skipped (resumed, {len(coref_links)} coref links)")

        # ── Phase 8 ─────────────────────────────────────────────────────
        if rfp <= 8:
            reason = "complex doc" if self._is_complex else "dead weight"
            print(f"\n[Phase 8] Implicit References — SKIPPED ({reason})")

            partial_links = self._inject_partial_references(
                sentences, components, name_to_id, transarc_set,
                {(c.sentence_number, c.component_id) for c in validated},
                {(l.sentence_number, l.component_id) for l in coref_links},
                set(),
            )
            if partial_links:
                print(f"\n[Phase 8b] Partial Injection")
                print(f"  Injected: {len(partial_links)} candidates")

            entity_links = [
                SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
                for c in validated
            ]
            all_links = transarc_links + entity_links + coref_links + partial_links
            link_map = {}
            for lk in all_links:
                key = (lk.sentence_number, lk.component_id)
                if key not in link_map:
                    link_map[key] = lk
                else:
                    old_p = self.SOURCE_PRIORITY.get(link_map[key].source, 0)
                    new_p = self.SOURCE_PRIORITY.get(lk.source, 0)
                    if new_p > old_p:
                        link_map[key] = lk
            preliminary = list(link_map.values())

            # Parent-overlap guard
            if not self._disable_parent_overlap and self.model_knowledge and self.model_knowledge.impl_to_abstract:
                child_to_parent = self.model_knowledge.impl_to_abstract
                sent_comps = defaultdict(set)
                for lk in preliminary:
                    sent_comps[lk.sentence_number].add(lk.component_name)
                before_po = len(preliminary)
                filtered_po = []
                for lk in preliminary:
                    parent = child_to_parent.get(lk.component_name)
                    if parent and parent in sent_comps[lk.sentence_number]:
                        print(f"    Parent-overlap drop: S{lk.sentence_number} -> {lk.component_name} "
                              f"(parent: {parent})")
                    else:
                        filtered_po.append(lk)
                if len(filtered_po) < before_po:
                    print(f"  Parent-overlap guard: dropped {before_po - len(filtered_po)}")
                preliminary = filtered_po

            # Phase 8c boundary filters
            print("\n[Phase 8c] Boundary Filters (non-TransArc only)")
            preliminary, boundary_rejected = self._apply_boundary_filters(
                preliminary, sent_map, transarc_set
            )
            if boundary_rejected:
                print(f"  Rejected: {len(boundary_rejected)}")

            self._save_phase(text_path, "pre_judge", {
                "preliminary": preliminary,
                "transarc_set": transarc_set,
            })
        else:
            data = self._load_phase(text_path, "pre_judge") or self._load_phase(text_path, "pre_judge", variant=self._checkpoint_source)
            preliminary = data["preliminary"]
            transarc_set = data["transarc_set"]
            print(f"\n[Phase 8] — skipped (resumed, {len(preliminary)} preliminary)")

        # ── Phase 9 ─────────────────────────────────────────────────────
        print("\n[Phase 9] Judge Review (TransArc immune)")
        reviewed = self._judge_review(preliminary, sentences, components, sent_map, transarc_set)
        rejected = [l for l in preliminary if (l.sentence_number, l.component_id)
                    not in {(r.sentence_number, r.component_id) for r in reviewed}]
        print(f"  Approved: {len(reviewed)} (rejected {len(rejected)})")

        # ── Phase 10 ────────────────────────────────────────────────────
        print("\n[Phase 10] FN Recovery — SKIPPED (dead weight)")
        final = reviewed

        # ── Save log + final checkpoint ──────────────────────────────────
        self._log("summary", {"total_time_s": round(time.time() - t0, 1)},
                  {"final": len(final)}, final)
        self._save_log(text_path)
        self._save_phase(text_path, "final", {
            "final": final, "reviewed": reviewed, "rejected": rejected,
        })

        print(f"\nFinal: {len(final)} links")
        return final

    # ── Phase 4: ILinker2 seed ───────────────────────────────────────────

    def _process_transarc(self, transarc_csv, id_to_name, sent_map, name_to_id):
        raw_links = self._ilinker2.link(self._cached_text_path, self._cached_model_path)
        result = []
        for lk in raw_links:
            result.append(SadSamLink(
                sentence_number=lk.sentence_number,
                component_id=lk.component_id,
                component_name=lk.component_name,
                confidence=lk.confidence,
                source="transarc",
            ))
        return result

    # ── Phase 3: Few-shot judge + CamelCase override ──────────────────────

    def _learn_document_knowledge_enriched(self, sentences, components):
        """Phase 3: Few-shot calibrated judge + CamelCase deterministic override."""
        comp_names = [c.name for c in components]
        doc_lines = [s.text for s in sentences[:150]]

        prompt1 = f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names)}

WHAT TO FIND:
1. ABBREVIATIONS: Short forms explicitly introduced in the document.
   Rule: The abbreviation must be defined in the text, e.g., "Full Name (FN)" introduces FN.
   Like "Abstract Syntax Tree (AST)" defines AST — look for the same parenthetical pattern.

2. SYNONYMS: Alternative names that SPECIFICALLY refer to one component.
   Rule: The alternative name must unambiguously identify exactly ONE component.
   APPROVE: A proper name, role title, or technical alias used interchangeably with the component
   REJECT: A generic description that could apply to anything (like "the system" or "the process")

3. PARTIAL REFERENCES: A shorter form of a multi-word component name used alone.
   Rule: A trailing word from a multi-word name that, in this document, consistently means the full name.
   APPROVE: Only if the short form is unambiguous — no other component shares this word
   REJECT: Common words that have ordinary English meanings beyond the component

DOCUMENT:
{chr(10).join(doc_lines)}

Return JSON:
{{
  "abbreviations": {{"short_form": "FullComponent"}},
  "synonyms": {{"specific_alternative_name": "FullComponent"}},
  "partial_references": {{"partial_name": "FullComponent"}}
}}
JSON only:"""

        data1 = self.llm.extract_json(self.llm.query(prompt1, timeout=150))

        all_mappings = {}
        if data1:
            for short, full in data1.get("abbreviations", {}).items():
                if full in comp_names:
                    all_mappings[short] = ("abbrev", full)
            for syn, full in data1.get("synonyms", {}).items():
                if full in comp_names:
                    all_mappings[syn] = ("synonym", full)
            for partial, full in data1.get("partial_references", {}).items():
                if full in comp_names:
                    all_mappings[partial] = ("partial", full)

        if all_mappings:
            mapping_list = [f"'{k}' -> {v[1]} ({v[0]})" for k, v in list(all_mappings.items())[:25]]

            prompt2 = f"""JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{chr(10).join(mapping_list)}

EXAMPLES — study these to calibrate your judgment:

Example 1 — APPROVE (proper name in context):
  'Scheduler' -> TaskScheduler (partial)
  Document says: "The Scheduler assigns threads to available cores."
  Verdict: APPROVE. "Scheduler" is capitalized mid-sentence and used as a proper
  name for the TaskScheduler component, not as a generic concept.

Example 2 — APPROVE (CamelCase identifier):
  'RenderEngine' -> GameRenderEngine (synonym)
  Document says: "The RenderEngine processes draw calls each frame."
  Verdict: APPROVE. CamelCase is a constructed identifier — it is a proper name,
  not a generic English word.

Example 3 — APPROVE (abbreviation with document evidence):
  'AST' -> AbstractSyntaxTree (abbrev)
  Document says: "The Abstract Syntax Tree (AST) represents the parsed program."
  Verdict: APPROVE. Explicitly defined in the document with parenthetical pattern.

Example 4 — REJECT (generic concept, not a component name):
  'process' -> OrderProcessor (partial)
  Document says: "The system will process incoming requests."
  Verdict: REJECT. "process" is used as a verb in its ordinary English sense,
  not as a name for OrderProcessor.

Example 5 — APPROVE (distinctive partial used as proper name):
  'Dispatcher' -> EventDispatcher (partial)
  Document says: "When an event arrives, the Dispatcher routes it to handlers."
  Verdict: APPROVE. "Dispatcher" is capitalized mid-sentence and refers
  specifically to EventDispatcher — it is a distinctive term in this document.

Example 6 — REJECT (ambiguous, refers to the whole system):
  'system' -> PaymentSystem (partial)
  Document says: "The system handles all transactions."
  Verdict: REJECT. "system" refers to the overall system, not specifically
  to PaymentSystem.

NOW JUDGE THE PROPOSED MAPPINGS. Apply these rules:

REJECT if ANY of these are true:
- The term is used in its ordinary English sense, NOT as a name for the component
- The term refers to a different component or to the system as a whole
- The mapping cannot be verified from the actual document text

APPROVE if ANY of these are true:
- The term is a CamelCase identifier (mixed lower-then-upper like "PaymentGateway")
  — CamelCase terms are constructed proper names, not generic English
- The term appears capitalized mid-sentence in the document — this signals
  proper name usage (e.g., "The Optimizer runs" uses Optimizer as a name)
- The term is used AS A NAME for the component in context, refers to exactly
  one component, and this can be verified from the document

Return JSON:
{{
  "approved": ["term1", "term2"],
  "generic_rejected": ["generic_term1"]
}}
JSON only:"""

            data2 = self.llm.extract_json(self.llm.query(prompt2, timeout=120))
            approved = set(data2.get("approved", [])) if data2 else set(all_mappings.keys())
            generic_terms = set(data2.get("generic_rejected", [])) if data2 else set()

            # ── Deterministic overrides on LLM-rejected terms ─────────
            if self._enable_camelcase:
                for term in list(generic_terms):
                    if re.search(r'[a-z][A-Z]', term):
                        generic_terms.discard(term)
                        approved.add(term)
                        print(f"    CamelCase override (rescued): {term}")

            if self._enable_uppercase:
                for term in list(generic_terms):
                    if term.isupper() and len(term) <= 4 and term in all_mappings:
                        generic_terms.discard(term)
                        approved.add(term)
                        print(f"    Uppercase override (rescued): {term}")

            if self._enable_v24:
                for term in list(generic_terms):
                    if term not in all_mappings:
                        continue
                    if any(term.lower() == cn.lower() for cn in comp_names):
                        generic_terms.discard(term)
                        approved.add(term)
                        print(f"    Exact-component override (rescued): {term}")
                    elif term[0].isupper() and ' ' in term:
                        generic_terms.discard(term)
                        approved.add(term)
                        print(f"    Multi-word proper-name override (rescued): {term}")

        else:
            approved = set()
            generic_terms = set()

        knowledge = DocumentKnowledge()
        knowledge.generic_terms = generic_terms

        for term, (typ, comp) in all_mappings.items():
            if term in approved:
                if typ == "abbrev":
                    knowledge.abbreviations[term] = comp
                    print(f"    Abbrev: {term} -> {comp}")
                elif typ == "synonym":
                    knowledge.synonyms[term] = comp
                    print(f"    Syn: {term} -> {comp}")
                else:
                    knowledge.partial_references[term] = comp
                    print(f"    Partial: {term} -> {comp}")

        if generic_terms:
            print(f"    Generic (rejected): {', '.join(list(generic_terms)[:5])}")

        # Deterministic CamelCase-split synonym injection (universal SE convention)
        for comp in [c.name for c in components]:
            split = re.sub(r'([a-z])([A-Z])', r'\1 \2', comp)
            split = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', split)
            if split != comp and split not in knowledge.synonyms:
                knowledge.synonyms[split] = comp
                print(f"    CamelCase syn: {split} -> {comp}")

        return knowledge

    # ── Phase 3b: Multiword partial enrichment (configurable threshold) ───

    def _enrich_multiword_partials(self, sentences, components):
        if not self.doc_knowledge:
            return
        added = []
        for comp in components:
            parts = comp.name.split()
            if len(parts) < 2:
                continue
            last_word = parts[-1]
            if len(last_word) < 4:
                continue
            last_lower = last_word.lower()
            other_match = any(
                c.name != comp.name and c.name.lower().endswith(last_lower)
                for c in components
            )
            if other_match:
                continue
            if last_lower in {s.lower() for s in self.doc_knowledge.synonyms}:
                continue
            if last_lower in {p.lower() for p in self.doc_knowledge.partial_references}:
                continue
            is_generic_word = last_lower in self.GENERIC_PARTIALS
            full_lower = comp.name.lower()
            mention_count = 0
            for sent in sentences:
                sl = sent.text.lower()
                if last_lower in sl and full_lower not in sl:
                    if is_generic_word:
                        cap_word = last_word[0].upper() + last_word[1:]
                        if re.search(rf'\b{re.escape(cap_word)}\b', sent.text):
                            mention_count += 1
                    else:
                        if re.search(rf'\b{re.escape(last_word)}\b', sent.text, re.IGNORECASE):
                            mention_count += 1
            if mention_count >= self._partial_min_count:
                self.doc_knowledge.partial_references[last_word] = comp.name
                added.append(f"{last_word} -> {comp.name} ({mention_count} caps mentions)")
        if added:
            print(f"\n[Phase 3b] Multi-word Enrichment (threshold={self._partial_min_count})")
            for a in added:
                print(f"    Auto-partial: {a}")

    # ── Phase 5b: Targeted extraction with dotted-path prompt ────────────

    def _targeted_extraction(self, unlinked_components, sentences, name_to_id, sent_map,
                              components=None, transarc_links=None, entity_candidates=None):
        if not unlinked_components:
            return []

        parent_map = {}
        existing_sent_comp = defaultdict(set)
        if components and transarc_links:
            all_comp_names = {c.name for c in components}
            for comp in unlinked_components:
                parents = set()
                for other_name in all_comp_names:
                    if other_name != comp.name and len(other_name) >= 3 and other_name in comp.name:
                        parents.add(other_name)
                if parents:
                    parent_map[comp.name] = parents
            for lk in transarc_links:
                existing_sent_comp[lk.sentence_number].add(lk.component_name)
            if entity_candidates:
                for c in entity_candidates:
                    existing_sent_comp[c.sentence_number].add(c.component_name)

        all_extra = []
        for comp in unlinked_components:
            if len(sentences) <= 60:
                doc_sents = sentences
            else:
                doc_sents = sentences[:30] + sentences[-30:]

            doc_text = "\n".join([f"S{s.number}: {s.text}" for s in doc_sents])

            aliases = []
            if self.doc_knowledge:
                for a, c in self.doc_knowledge.abbreviations.items():
                    if c == comp.name: aliases.append(a)
                for s, c in self.doc_knowledge.synonyms.items():
                    if c == comp.name: aliases.append(s)
                for p, c in self.doc_knowledge.partial_references.items():
                    if c == comp.name: aliases.append(p)

            alias_str = f"\nKNOWN ALIASES: {', '.join(aliases)}" if aliases else ""

            prompt = f"""Find ALL sentences that discuss the software component "{comp.name}".
{alias_str}

Look for:
- Direct mentions of "{comp.name}" or any alias
- Descriptions of what {comp.name} does (functional descriptions)
- References to {comp.name}'s role in the architecture

Exclude:
- Names that appear only inside a dotted package path (e.g., com.example.name does NOT count as a reference to "name")

DOCUMENT:
{doc_text}

Return JSON:
{{"references": [{{"sentence": N, "matched_text": "text found", "reason": "why this refers to {comp.name}"}}]}}

Be thorough — find ALL sentences that discuss this component.
JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
            if not data:
                continue

            for ref in data.get("references", []):
                snum = ref.get("sentence")
                if not snum:
                    continue
                try:
                    snum = int(snum)
                except (ValueError, TypeError):
                    continue
                sent = sent_map.get(snum)
                if not sent:
                    continue
                cid = name_to_id.get(comp.name)
                if not cid:
                    continue

                if comp.name in parent_map:
                    parents_here = parent_map[comp.name] & existing_sent_comp.get(snum, set())
                    if parents_here:
                        print(f"    Targeted skip (parent overlap): S{snum} -> {comp.name} "
                              f"(parent: {', '.join(parents_here)})")
                        continue

                matched = ref.get("matched_text", comp.name)
                all_extra.append(CandidateLink(
                    snum, sent.text, comp.name, cid,
                    matched, 0.85, "entity", "targeted", True
                ))
                print(f"    Targeted: S{snum} -> {comp.name} ({matched})")

        return all_extra
