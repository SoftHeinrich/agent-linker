"""AgentLinker V9: Multi-run majority voting.

Runs the V6 pipeline N times (default 3), then majority-votes on
(sentence, component) pairs. A link is accepted if >= ceil(N/2) runs agree.

This eliminates outlier FP clusters caused by LLM variance (e.g. validated
GAE Datastore explosions on teammates, extra implicit FPs on teastore)
while preserving links that the pipeline reliably produces.

Results: 92.2% macro F1 (vs V6 mean 91.4%), with much lower variance.
Cost: ~3x a single V6 run.
"""

import math
import time
from collections import Counter
from typing import Optional

from ...core import SadSamLink
from ...llm_client import LLMBackend
from .agent_linker_v6 import AgentLinkerV6


class AgentLinkerV9(AgentLinkerV6):
    """Multi-run majority voting over the full V6 pipeline."""

    def __init__(self, backend: Optional[LLMBackend] = None,
                 post_filter: str = "none", n_runs: int = 3):
        super().__init__(backend=backend, post_filter=post_filter)
        self.n_runs = n_runs
        self.vote_threshold = math.ceil(n_runs / 2)
        print(f"AgentLinkerV9: {n_runs}-run majority vote, threshold={self.vote_threshold}")

    def link(self, text_path: str, model_path: str,
             transarc_csv: str = None) -> list[SadSamLink]:

        all_run_links = []
        run_details = []

        for run_idx in range(self.n_runs):
            print(f"\n{'='*80}")
            print(f"VOTE RUN {run_idx + 1}/{self.n_runs}")
            print(f"{'='*80}")

            t0 = time.time()
            links = super().link(text_path, model_path, transarc_csv)
            elapsed = time.time() - t0

            link_set = {(l.sentence_number, l.component_id): l for l in links}
            all_run_links.append(link_set)
            run_details.append({
                "run": run_idx + 1,
                "n_links": len(links),
                "time": elapsed,
            })
            print(f"  Run {run_idx + 1}: {len(links)} links ({elapsed:.0f}s)")

        # Majority vote
        vote_counts = Counter()
        link_pool = {}
        for link_set in all_run_links:
            for key, link in link_set.items():
                vote_counts[key] += 1
                if key not in link_pool:
                    link_pool[key] = link

        final = []
        for key, count in vote_counts.items():
            if count >= self.vote_threshold:
                link = link_pool[key]
                final.append(SadSamLink(
                    link.sentence_number, link.component_id,
                    link.component_name,
                    count / self.n_runs,
                    link.source,
                ))

        accepted = len(final)
        rejected = len(vote_counts) - accepted
        print(f"\n{'='*80}")
        print(f"VOTE RESULT: {accepted} accepted, {rejected} rejected "
              f"(threshold: {self.vote_threshold}/{self.n_runs})")
        for detail in run_details:
            print(f"  Run {detail['run']}: {detail['n_links']} links ({detail['time']:.0f}s)")
        print(f"  Union: {len(vote_counts)}, Final: {accepted}")
        print(f"{'='*80}")

        return final
