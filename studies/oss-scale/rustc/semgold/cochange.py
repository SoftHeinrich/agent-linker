"""P2 co-change gold at sentence level, mined from the rust monorepo history.

A commit that edits a guide chapter AND code under 1..MAX_CRATES compiler crates is a
maintainer's assertion that the edited prose and those crates are related. The chapter's
diff (post-image lines) is matched to today's sentences; a sentence that a focused commit
added or rewrote receives the commit's crates. Independent of the sentence's wording.
"""
from __future__ import annotations

import collections
import json
import re
import subprocess
import sys

from common import DATA, GUIDE_PREFIX, OUT, RUST, load_components, load_sentences, norm, tokens, write_csv

MAX_CRATES = 3          # more = a bulk refactor, not a focused change
MIN_SENT_TOKENS = 6     # avoid trivially matching short sentences
JACCARD = 0.8


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(RUST), *args], capture_output=True, text=True, check=True).stdout


def commits_touching(path: str) -> list[tuple[str, str]]:
    out = git("log", "--no-merges", "--follow", "--format=%H %ad", "--date=short", "--", path)
    return [tuple(line.split(" ", 1)) for line in out.splitlines() if line.strip()]


def commit_files(sha: str) -> list[str]:
    return [l for l in git("show", "--format=", "--name-only", sha).splitlines() if l.strip()]


def added_lines(sha: str, path: str) -> list[str]:
    try:
        diff = git("show", "--format=", "-U0", sha, "--", path)
    except subprocess.CalledProcessError:
        return []
    return [l[1:] for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]


def main() -> None:
    crates = set(load_components())
    rows = load_sentences()
    by_chapter = collections.defaultdict(list)
    for r in rows:
        by_chapter[r["chapter"]].append(r)
    pairs = []
    chapter_pairs = []
    stats = collections.Counter()
    for chapter, sents in sorted(by_chapter.items()):
        path = GUIDE_PREFIX + chapter
        for sha, date in commits_touching(path):
            files = commit_files(sha)
            touched = {f.split("/")[1] for f in files if f.startswith("compiler/") and f.count("/") >= 2}
            touched &= crates
            stats["commits"] += 1
            if not touched:
                stats["doc_only"] += 1
                continue
            if len(touched) > MAX_CRATES:
                stats["unfocused"] += 1
                continue
            stats["focused"] += 1
            for c in sorted(touched):
                chapter_pairs.append({"chapter": chapter, "crate": c, "commit": sha, "date": date})
            added = added_lines(sha, path)
            added_norm = [norm(l) for l in added if l.strip()]
            blob = " ".join(added_norm)
            hit = 0
            for s in sents:
                st = tokens(s["text"])
                if len(st) < MIN_SENT_TOKENS:
                    continue
                sn = " ".join(st)
                match = sn in blob
                if not match:
                    sset = set(st)
                    for line in added_norm:
                        lt = set(line.split())
                        if lt and len(sset & lt) / len(sset | lt) >= JACCARD:
                            match = True
                            break
                if match:
                    hit += 1
                    for c in sorted(touched):
                        pairs.append({"sentence": s["number"], "crate": c, "commit": sha, "date": date,
                                      "chapter": chapter, "n_crates": len(touched)})
            stats["sentence_hits"] += hit
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "cochange_pairs.csv", pairs, ["sentence", "crate", "commit", "date", "chapter", "n_crates"])
    write_csv(OUT / "cochange_chapter_pairs.csv", chapter_pairs, ["chapter", "crate", "commit", "date"])
    uniq = {(p["sentence"], p["crate"]) for p in pairs}
    summary = {
        "commits_touching_core_chapters": stats["commits"],
        "doc_only": stats["doc_only"], "unfocused_gt_max_crates": stats["unfocused"], "focused": stats["focused"],
        "sentence_crate_pairs": len(uniq), "sentences_with_pair": len({p["sentence"] for p in pairs}),
        "crates_with_pair": len({p["crate"] for p in pairs}),
        "chapter_crate_pairs": len({(p["chapter"], p["crate"]) for p in chapter_pairs}),
        "top_crates": collections.Counter(c for _, c in uniq).most_common(15),
    }
    (OUT / "cochange_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
