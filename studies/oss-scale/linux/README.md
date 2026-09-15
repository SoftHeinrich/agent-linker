# Auditing the semantic-gold recipe against a human mapping (Linux `MAINTAINERS`)

The label model in `../README.md` §8 is built by two LLM families and checked by a third
model. Its open item was human confirmation. The Linux kernel supplies one for free:
`MAINTAINERS` is a component registry written and kept current by the maintainers
themselves, and **1,715 of its 3,395 entries claim `Documentation/` files alongside their
code paths**. A documentation file listed under a subsystem is a human-made
documentation→component trace link — just not called one.

This audits the recipe on that mapping: foreign system, foreign language, no rustc
carry-over, and an assignment nobody made for a traceability paper.

## Set-up

```bash
curl -s https://raw.githubusercontent.com/torvalds/linux/master/MAINTAINERS -o /tmp/MAINTAINERS
python3 build.py --docs 12                      # sample, fetch, sentence-split
OPENAI_API_KEY="$OAI_KEY" OPENAI_SERVICE_TIER=flex OPENAI_REASONING_EFFORT=none \
  python3 annotate.py --backend openai --model gpt-5.6-terra --workers 6 --tag terra
python3 score.py --tag terra
```

* Components: the 3,282 `MAINTAINERS` entries that have code paths. Profile = entry title
  + its code paths (nothing hand-written for this study).
* Documents: of the 607 documentation files claimed by **exactly one** subsystem, 12 sampled
  at seed 0 with 15–60 sentences: 383 sentences over filesystems, hwmon, gpu, mm, media,
  tee, wmi, dev-tools.
* Prompt: the §8 ABOUT/REFERS rubric, 12 BM25 candidates, ±2 sentences of context. The
  annotator is never told which subsystem owns the file. The owner is forced into the
  candidate list so the check measures labelling rather than retrieval; retrieval is
  reported on its own.

## Result

| | |
|---|---|
| sentences labelled | 380 (41 got no ABOUT at all) |
| **owner among the ABOUT labels** | **298/380 = 0.784** |
| of the sentences that got any ABOUT | 298/339 = 0.879 |
| owner labelled REFERS instead | 14 |
| ABOUT labels per sentence | 0.93 of 12 candidates — not a yes-to-everything labeller |
| **documents whose most-voted subsystem is the human owner** | **10/12** |

Per document (`out/score_terra.txt`): 9p 34/38, hns3-pmu 30/30, dma-buf-heaps 18/19,
uniwill-laptop 38/41, container 43/49, yogafan 16/18, smsc47b397 24/28, hugetlb 39/47,
camera-sensor 32/45, ts-tee 13/19, **afbc 5/16**, **dlmfs 6/30**.

The two failures are both worth keeping:

* `Documentation/gpu/afbc.rst` is owned by `ARM MALI-DP DRM DRIVER`; the text describes the
  AFBC buffer format itself, and the annotator voted `THE REST` (the kernel's catch-all
  entry). Ownership here is organisational, and the doc is genuinely not about the driver.
* `Documentation/filesystems/dlmfs.rst` is owned by `ORACLE CLUSTER FILESYSTEM 2 (OCFS2)`
  because dlmfs lives in the ocfs2 tree; the annotator voted `DISTRIBUTED LOCK MANAGER (DLM)`,
  which is what the file is about. The human mapping is a code-location fact, the label is a
  content fact, and they disagree.

So the human source is itself not a pure aboutness gold — it is *who maintains the file*.
That is the expected direction of error, and it bounds this audit rather than invalidating it.

## The finding that transfers back to the linker

Retrieval, not labelling, is the bottleneck at this universe size:

| candidate query | owner in top-12 of 3,282 |
|---|---|
| per 5-sentence batch | 0.346 (median rank 25) |
| pooled over the whole document | **0.750** |

Same index, same components — only the query changes. The linker's per-sentence stages are
searching with a query that is too small, which is the same shape as the rustc partial-name
collapse (P 0.07 at 79 components). Candidates should be retrieved per document and judged
per sentence.

## Files

`build.py` parse + sample · `annotate.py` candidates + prompt + cached LLM call ·
`score.py` agreement with the human owner · `out/` dataset, annotations, scores.
LLM responses reuse `../rustc/semgold/cache`, so the run reproduces without a key.
