"""AgentLinker V8a: Union-accept voting with tiebreaker judge.

Key insight from voting experiments:
- Majority vote (S1) kills borderline TPs that appear in only 1/3 runs → FN increase
- Oracle union shows 95.9% ceiling (union TPs, intersect FPs)
- Solution: take UNION of all runs, then apply strict tiebreaker judge ONLY to
  unstable links (those appearing in <threshold runs)

Approach:
1. Run V6 pipeline N times (default 3)
2. Accept all links appearing in >= threshold runs (stable = likely correct)
3. For links in <threshold runs (unstable), run a tiebreaker judge that decides
   whether to keep or reject each one
4. This preserves recall (union baseline) while filtering unstable FPs
"""

import math
import os
import re
import time
from collections import Counter
from typing import Optional

from ...core import SadSamLink, DocumentLoader
from ...pcm_parser import parse_pcm_repository
from ...llm_client import LLMBackend
from .agent_linker_v6 import AgentLinkerV6


class AgentLinkerV8a(AgentLinkerV6):
    """Union-accept voting: take union of N runs, tiebreaker judge for unstable links."""

    def __init__(self, backend: Optional[LLMBackend] = None,
                 post_filter: str = "none", n_runs: int = 3):
        super().__init__(backend=backend, post_filter=post_filter)
        self.n_runs = n_runs
        self.stable_threshold = math.ceil(n_runs / 2)  # >= 2/3 = stable
        print(f"  V8a union-judge: {n_runs} runs, stable={self.stable_threshold}/{n_runs}")

    def link(self, text_path: str, model_path: str,
             transarc_csv: str = None) -> list[SadSamLink]:

        # Load data for tiebreaker judge context
        components = parse_pcm_repository(model_path)
        sentences = DocumentLoader.load_sentences(text_path)
        sent_map = DocumentLoader.build_sent_map(sentences)
        comp_names = [c.name for c in components]

        all_run_links = []
        for run_idx in range(self.n_runs):
            print(f"\n{'='*80}")
            print(f"UNION-JUDGE RUN {run_idx + 1}/{self.n_runs}")
            print(f"{'='*80}")

            t0 = time.time()
            links = super().link(text_path, model_path, transarc_csv)
            elapsed = time.time() - t0

            link_set = {(l.sentence_number, l.component_id): l for l in links}
            all_run_links.append(link_set)
            print(f"  Run {run_idx + 1}: {len(links)} links ({elapsed:.0f}s)")

        # Count votes per link
        vote_counts = Counter()
        link_pool = {}
        source_pool = {}
        for link_set in all_run_links:
            for key, link in link_set.items():
                vote_counts[key] += 1
                if key not in link_pool:
                    link_pool[key] = link
                    source_pool[key] = link.source

        # Split into stable and unstable
        stable = []
        unstable = []
        for key, count in vote_counts.items():
            link = link_pool[key]
            if count >= self.stable_threshold:
                stable.append(link)
            else:
                unstable.append(link)

        print(f"\n{'='*60}")
        print(f"VOTE SPLIT: {len(stable)} stable (>={self.stable_threshold}/{self.n_runs}), "
              f"{len(unstable)} unstable (<{self.stable_threshold}/{self.n_runs})")

        # Tiebreaker judge for unstable links
        if unstable:
            kept = self._tiebreaker_judge(unstable, vote_counts, comp_names, sent_map)
            print(f"  Tiebreaker: {len(kept)}/{len(unstable)} unstable links kept")
        else:
            kept = []

        final = stable + kept

        print(f"FINAL: {len(final)} links (={len(stable)} stable + {len(kept)} tiebroken)")
        print(f"{'='*60}")

        return final

    def _tiebreaker_judge(self, unstable_links, vote_counts, comp_names, sent_map):
        """Strict judge for links that appeared in only some runs.

        These are borderline cases — they might be valid TPs that some runs
        missed, or they might be random FPs from one noisy run. Use a judge
        with extra context (vote count) to decide.
        """
        if not unstable_links:
            return []

        cases = []
        for i, link in enumerate(unstable_links[:30]):
            sent = sent_map.get(link.sentence_number)
            prev = sent_map.get(link.sentence_number - 1)
            key = (link.sentence_number, link.component_id)
            votes = vote_counts[key]

            ctx_lines = []
            if prev:
                ctx_lines.append(f"    PREV: {prev.text[:60]}...")
            ctx_lines.append(f"    >>> S{link.sentence_number}: {sent.text if sent else '?'}")

            cases.append(
                f"Case {i+1}: S{link.sentence_number} -> {link.component_name} "
                f"(src:{link.source}, votes:{votes}/{self.n_runs})\n"
                + chr(10).join(ctx_lines)
            )

        prompt = f"""TIEBREAKER JUDGE: These trace links appeared in only SOME of {self.n_runs} independent analysis runs. Links that appear in more runs are more likely correct.

COMPONENTS: {', '.join(comp_names)}

Your task: Approve links that are genuinely about the component. Reject links where:
- The component name is used as a generic English word (not an architecture reference)
- The sentence describes package/directory structure using dot notation
- The link is based on a technology reference, not the architectural component
- A pronoun resolution seems incorrect or ambiguous
- The component name appears only as a modifier or in a compound expression

LINKS:
{chr(10).join(cases)}

Return JSON:
{{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=180))
        if not data:
            return []  # Conservative: reject all if judge fails

        kept = []
        for j in data.get("judgments", []):
            idx = j.get("case", 0) - 1
            if 0 <= idx < len(unstable_links[:30]) and j.get("approve", False):
                kept.append(unstable_links[idx])

        # Links beyond the 30-case limit: accept if votes > 1
        for link in unstable_links[30:]:
            key = (link.sentence_number, link.component_id)
            if vote_counts[key] > 1:
                kept.append(link)

        return kept
