"""Fixture loader for `pilot/test_s126.py`.

Loads a project's sentences/components/gold plus the pinned `knowledge` and
`linker_name` phase states from a recorded run, so the s126 contract checks
resample only the union judge and the coreference resolver.

The stage-pilot comparison this module used to run (`head`/`annotexact`/
`refuse` arms against the retired `s_linker123`) already answered its
question -- the antecedent-form contract it measured is now enforced in
`s_linker126` itself (`_written_as` / `ANTECEDENT_FORMS`). See
`origin/archive/master-pre-s126-consolidation` for that pilot's arms and
its recorded numbers.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from llm_sad_sam.core.document_loader_v2 import build_sent_map, load_sentences  # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository  # noqa: E402

from reading_pilots import BENCH, DATASETS, gold_pairs  # noqa: E402

DEFAULT_RUN = ROOT.parent / "results/shortlistmark_e2e_terra_r1_20260914"


def phase(run: Path, project: str, name: str, variant="s_linker123"):
    path = run / "phase_states" / variant / "openai" / project / f"{name}.pkl"
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load(project, run):
    text, repo, gold_path = DATASETS[project]
    components = parse_pcm_repository(str(BENCH / repo))
    sentences = load_sentences(str(BENCH / text))
    name_state = phase(run, project, "linker_name")
    return {
        "components": components,
        "sentences": sentences,
        "sent_map": build_sent_map(sentences),
        "name_to_id": {c.name: c.id for c in components},
        "gold": gold_pairs(BENCH / gold_path),
        "knowledge": phase(run, project, "knowledge")["doc_knowledge"],
        "name_links": {(l.sentence_number, l.component_id)
                       for l in name_state["links"]},
    }
