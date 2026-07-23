# Spike 005 — Results

## Step 1 ($0) — candidate-pool recall ceilings

Harness: `harness/candidate_gap.py` (pure cache read, no LLM). Pool = entity candidates ∪
coref raw, before validation; a validator can only approve pool members.

| dataset | nothink ceil | thinking-on ceil | gap (pts) |
|---------|--------------|------------------|-----------|
| mediastore | 92.5 | 98.9 | +6.5 |
| teastore | 98.8 | 100.0 | +1.2 |
| teammates | 88.9 | 91.8 | +2.9 |
| bigbluebutton | 72.0 | 83.3 | +11.3 |
| jabref | 100.0 | 100.0 | 0.0 |
| **MACRO** | **90.4** | **94.8** | **+4.4** |

**Gold-link decomposition (585 instances over 15 cells):**
- **validator-recoverable** (in nothink's pool): **506 = 86.5%** — spike 004's lever.
- **extraction-bound** (thinking-on pool only): **36 = 6.2%** — this spike's lever
  (= the ~1.1 upstream F1 pts; +4.4 recall-ceiling pts).
- **unreachable** (neither pool): 43 = 7.4% — the true recall floor.

**The upstream lever is small and concentrated.** Of the 36 extraction-bound true links:
- **bigbluebutton: 15** — almost all the same component ("HTML5 Client"); nothink's
  effort-0 extraction never proposes it. bbb's ceiling gap (+11.3) is mostly this.
- **mediastore: 5** (FileStorage/DB/MediaAccess), **teammates: 4** (Logic/Storage/GAE),
  **teastore: 1** (WebUI), **jabref: 0**.

**Read:** the residual ~1.1 upstream macro pts come from ~36 specific true links that
no-reasoning extraction/coref-discovery fails to even propose — dominated by one bbb
component. Recovering them needs thinking (or better prompting) at the EXTRACTION /
coref-discovery phase, not the validator. Because the lever is small (6.2% of gold),
bbb-dominated, and would require re-running the expensive extraction phase with thinking,
the cost/benefit of a full Step-2 extraction replay is marginal. The high-value, narrow
target is bbb's missing "HTML5 Client" mentions.

## Step 2 ($0, cache-only) — mechanism of the extraction-bound links

Question (user steer): can the **same reasoning-relocation mechanism** that recovered the
validation gates (spike 004's Mode-5 justification scaffold) also recover the extraction
miss? That hinges on *why* thinking-on surfaces these mentions. Harness
`harness/extraction_mechanism.py` reads the frozen thinking-on cache and, for every
extraction-bound gold link (in thinking-on's candidate pool, never in nothink's),
classifies HOW thinking-on surfaced it. Pure cache read, no LLM.

**Finding 1 — the gap is small AND ~2/3 run-variance.** Step 1 counted 36 extraction-bound
gold *instances* (summed over 15 cells). Those are only **25 distinct (sentence, component)
links**, and **17 of the 25 (68%) appear in just 1 of 3 thinking-on runs** — only 3 links
are robust 3/3, 5 are 2/3. So most of the "+1.1 upstream" is **thinking-on's own extraction
variance**, not a deterministic capability effort-0 lacks. Averaging more thinking-on runs
would recover about as much as any new mechanism.

**Finding 2 — mechanism split (25 distinct links):**

| mechanism | n | % | what it is |
|-----------|---|---|------------|
| literal | 8 | 32% | component name (or near-variant) is verbatim in the sentence; effort-0 just *skipped an explicit mention* (e.g. S4 "HTML5 client.", teammates S8 "The main logic of the application") |
| coref | 6 | 24% | surfaced only via anaphora chains (all of bbb's "each client" → "client side" → "the client side", each 1/3 runs) |
| **indirect** | **11** | **44%** | name NOT verbatim; role/participant **inference** (S9 "communication between client and server" → HTML5 Client; mediastore "datastore" → GAE Datastore; teastore "UI" → WebUI) |

bbb dominates (15/25): 5 literal, 6 coref (all the noisy 1/3 "client side" chain), 4 indirect
(two of which — S9, S79 HTML5 Client — are the robust 3/3 core of the whole gap).

**Finding 3 — the mechanism does NOT transfer to the dominant bucket.** Spike 004's Mode-5
trick relocates *discrimination* reasoning into output tokens — it improves a keep/reject
**decision about a candidate that already exists**. It cannot make the model *propose* a
non-verbatim mapping it never surfaced. So:
- **44% indirect** (the largest, and the robust core): needs the model to GENERATE
  "client and server" → HTML5 Client — exactly thinking-as-generator. A justification field
  over already-surfaced candidates can't force this. **Out of reach of the mechanism.**
- **24% coref**: needs an anaphora-resolution scaffold (a different tool), and is entirely
  noise-dominated (every one is 1/3 runs).
- **32% literal**: the only clean fit — a *recall-expansion* prompt ("never skip a sentence
  that names a component verbatim") could plausibly recover these at effort-0. But they too
  are mostly single-run; the robust, repeatable literal-skip deficit is ~2 links.

## Step 2 — VERDICT

**Can the same mechanism improve extraction? Mostly NO — and the prize is smaller and noisier
than the 1.1-pt headline.** There is a clean asymmetry:

> Thinking does **precision/discrimination** work at the validation gates (which
> reasoning-relocation reconstructs well — spike 004 shipped +1.1 Sonnet / +3.8 gpt) and
> **recall/generation** work at extraction (which a justification-field scaffold *cannot*
> reconstruct — you can't justify a candidate you never proposed). The Mode-5 mechanism
> substitutes for thinking-as-discriminator, not thinking-as-generator.

The addressable, *robust* extraction-bound signal is ~3–8 links, 44% of it genuine
non-verbatim inference the mechanism can't reach, and ~2/3 of the headline gap is thinking-on
run-variance. **Recommendation: do NOT run the bbb LLM extraction probe** — the $0 inspection
already shows the cost/benefit is poor. If any cheap follow-up is worth it, it's a single
effort-0 *recall-expansion* extraction prompt tested on the literal-skip subset only;
expected realizable macro gain is well under 1 pt and likely within noise. **Low priority.**

(Decision retained from the original Step-2 fork: "stop" — now backed by the mechanism data,
not just the cost estimate.)
