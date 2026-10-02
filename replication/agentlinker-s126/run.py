import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("NLTK_DATA", str(ROOT / "nltk_data"))

from llm_sad_sam.linkers.experimental.s_linker126 import SLinker126
from llm_sad_sam.llm_client import LLMBackend

DATA = {
    "mediastore": ("text_2016/mediastore.txt", "model_2016/pcm/ms.repository"),
    "teammates": ("text_2021/teammates.txt", "model_2021/pcm/teammates.repository"),
    "teastore": ("text_2020/teastore.txt", "model_2020/pcm/teastore.repository"),
    "bigbluebutton": ("text_2021/bigbluebutton.txt", "model_2021/pcm/bbb.repository"),
    "jabref": ("text_2021/jabref.txt", "model_2021/pcm/jabref.repository"),
}


def write_links(links, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sentence", "component_id", "component_name", "confidence", "source"])
        for link in sorted(links, key=lambda item: (item.sentence_number, item.component_id)):
            writer.writerow([
                link.sentence_number,
                link.component_id,
                link.component_name,
                f"{link.confidence:.2f}",
                link.source,
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("terra", "luna"), default="terra")
    parser.add_argument("--datasets", nargs="+", choices=DATA, default=list(DATA))
    parser.add_argument("--no-knowledge", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("run-output"))
    parser.add_argument("--from-cache", type=Path)
    args = parser.parse_args()

    if args.from_cache is None and not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is required for a live run")
    os.environ["OPENAI_MODEL_NAME"] = f"gpt-5.6-{args.model}"
    os.environ.setdefault("OPENAI_REASONING_EFFORT", "none")
    os.environ.setdefault("OPENAI_SERVICE_TIER", "flex")
    os.environ["PHASE_CACHE_DIR"] = str(args.output / "phase_states")
    os.environ["LLM_LOG_DIR"] = str(args.output / "llm_logs")
    variant = "s_linker126_noknow" if args.no_knowledge else "s_linker126"

    for dataset in args.datasets:
        if args.from_cache is None:
            text, model = DATA[dataset]
            base = ROOT / "data" / dataset
            linker = SLinker126(
                backend=LLMBackend.OPENAI,
                model=f"gpt-5.6-{args.model}",
                no_knowledge=args.no_knowledge,
            )
            links = linker.link(str(base / text), str(base / model))
        else:
            snapshot = args.from_cache / "phase_states" / "s_linker126" / "openai" / dataset / "final.pkl"
            with snapshot.open("rb") as handle:
                links = pickle.load(handle)["final"]
        output = args.output / f"{variant}_{dataset}_links.csv"
        write_links(links, output)
        print(f"{dataset}: {len(links)} links -> {output}")


if __name__ == "__main__":
    main()
