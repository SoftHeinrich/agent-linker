import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("NLTK_DATA", str(ROOT / "nltk_data"))
sys.path.insert(0, str(ROOT))

from agentlinker.linker import AgentLinker
from agentlinker.llm import OpenAIChat, ReplayChat

DATASETS = {
    "mediastore": ("text_2016/mediastore.txt", "model_2016/pcm/ms.repository"),
    "teammates": ("text_2021/teammates.txt", "model_2021/pcm/teammates.repository"),
    "teastore": ("text_2020/teastore.txt", "model_2020/pcm/teastore.repository"),
    "bigbluebutton": ("text_2021/bigbluebutton.txt", "model_2021/pcm/bbb.repository"),
    "jabref": ("text_2021/jabref.txt", "model_2021/pcm/jabref.repository"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("terra", "luna"), default="terra")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--no-aliases", action="store_true")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        if args.replay:
            chat = ReplayChat(args.replay / f"{dataset}_calls.json")
        else:
            chat = OpenAIChat(f"gpt-5.6-{args.model}")
        text, model = DATASETS[dataset]
        links = AgentLinker(chat, use_aliases=not args.no_aliases).link(
            ROOT / "data" / dataset / text, ROOT / "data" / dataset / model)
        if args.replay and chat.unused():
            raise SystemExit(f"{dataset}: {chat.unused()} recorded calls were not replayed")

        with open(args.output / f"{dataset}_links.csv", "w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["sentence", "component_id", "component_name", "source"])
            for link in sorted(links, key=lambda link: (link.sentence, link.component_id)):
                writer.writerow([link.sentence, link.component_id, link.component_name, link.source])
        if not args.replay:
            with open(args.output / f"{dataset}_calls.json", "w", encoding="utf-8") as handle:
                json.dump(chat.calls, handle, indent=1, ensure_ascii=False)
        print(f"{dataset}: {len(links)} links, {len(chat.calls)} calls")


if __name__ == "__main__":
    main()
