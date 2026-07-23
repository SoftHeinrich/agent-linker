# Open issues from the comprehensive review

Captured 2026-06-29 from the multi-agent paper review (claims/LaTeX/terminology/structure/writing).
Lists what is **still unresolved**. Mechanical fixes already done are in "Resolved" at the bottom.

Status tags: **[decide]** needs an authorial call · **[mech]** mechanical once a target is picked · **[chore]** housekeeping.

---

## Content / correctness — [decide]

- [ ] **Two-backend / Claude numbers pending** — `discussion.tex:22` (`%TODO` pin the exact Claude Sonnet version + fill numbers) and `results.tex:15` ("pending its run"). The two-backend robustness claim is not fully backed until the Claude numbers land; it is also absent from the abstract/intro. **BLOCKED on the Claude Sonnet run — cannot resolve in prose.**
- [ ] **Unresolved reviewer comments** — `intro.tex:80-92`: 8 carried-over Jan Keim comments, each marked "remove once resolved." **Verified 2026-06-29** (see below); 6/8 addressed, 2 borderline. Awaiting author OK to delete the block.
  - [P1 F1→decisions] addressed: `:15` now says "misses many true links / returns many wrong ones," no count claimed.
  - [P3 vs Artemis NAER] addressed: `:27` + `results.tex:42` state Artemis keeps no reusable knowledge.
  - [P4 "runtime knowledge layer"] addressed: intro now says "a knowledge module computed once" (jargon gone from intro body).
  - [P4 "surface match catch too much"] **borderline**: intro `:37` still terse; the *mechanism* is only shown later in `motivation.tex` (preferences example).
  - [P4 examples] addressed via the `motivation.tex` JabRef example.
  - [P4 no-tuning claim] **borderline**: `:47` still asserts "no labeled trace links / no tuning"; SWATTR/Artemis also need none, so it is not a differentiator — keep as a property, not a claimed advantage.
  - [P5 "ruler" term] addressed: removed; now "evaluation suite / metrics."
  - [P6 overall] non-actionable general comment.

## Plain-language pass (2026-06-29) — paraphrasing → plainest term

User directive: never paraphrase *links* as "pairs"; scan the whole doc and use the plainest
wording for every concept (collapse elegant variation).

**Applied (one plain term per concept):**
- **links** — removed every "pair" paraphrase: `link pairs` / `link-level pairs` / `sentence--file pair` / `sentence--component pair` / `candidate link (s,c) pair` / `a pair of linkers` → plain **links** / **two linkers** (intro, motivation, metric, approach, results captions). Kept the verb "paired with a judge" (intro:35) and the related-work IR term "artifact pair" (rw:5).
- **judges** — `evidence-grounded checks` / `each check` / `different checks` → **judges** (eval:42-43, approach:125), matching the canonical "judge" used everywhere else.
- **the metric** — `volume-weighted average` → **link-level/file \fone** (results, metric); `component-weighted view` → **the size-aware suite** (results); `the standard score` → **the standard metric** (intro). 
- **linker** — `the \linkerB extractor` → **\linkerB** (eval:96).
- **judge term unified** — `evidence-grounded` → **evidence-backed** everywhere, via new macro `\evidenceBacked` (abbrev.tex); body prose uses the macro, the two headings/prompt title stay literal "Evidence-Backed". (One commented-out line, intro:77, still reads "evidence-grounded" — not rendered.)

**Marked for your call (left as-is — possible meaning, not pure variation):**
- [ ] **`file \fone` vs `link-level \fone`** — the doc-code standard metric is called both ("file F1" in results, "link-level F1" generally). They're equal in doc-code (a link = sentence–file) but the split may be an intentional grain cue. Unify, or keep file=doc-code / link-level=general?
- [ ] **`the score`** — intro:58, motivation:98 ("drive the score" / "invisible to the score"). Plain already; left. Collapse to "link-level \fone" if you want one term only.
- [ ] **`architecture-model element`** vs **`component`** — results:119 (named judge "matches no architecture-model element"). Left in case "element" is broader than "component" (interfaces). Say the word if it should be "component".

## Still open — blocked or author-only

- [x] **Two-backend / Claude numbers** — RESOLVED 2026-06-29: the Claude Sonnet numbers are in the appendix mirror. Remaining micro-task: pin the exact Sonnet version string in the appendix caption (`discussion.tex:26` TODO slimmed to that).
- [ ] **Reviewer block deletion** — `intro.tex:80-92`. Verified 6/8 addressed, 2 borderline (see above). **Awaiting author OK to delete**; the two borderline ones (P4 "catch too much", P4 no-tuning) may want a one-line tweak first.
- [ ] **RQ3/RQ4 run-count** — `discussion.tex:12` + `results.tex:10` SPEC. Author call: keep RQ3/4 single-run (+ disclosure) or upgrade to mean-of-3. The within-run justification text hinges on this.
- [ ] **Generated tables — convention match** — the leading-zero + `pp` decisions were applied to **prose only**. `tab:rq1/rq2/rq4` are generated from the transarc-emp CSV→TeX pipeline; regenerate them so cell formatting matches (do not hand-edit `table/*.tex`).

---

## Resolved 2026-06-29 (pass 2 — after author decisions Q1–Q4)

- **Number format** — leading zeros everywhere (`0.84`, `0.71`, `0.75`…); all gains/deltas as `pp` (`+10.0`pp, `+5.7`pp, `+13/+41/+42`pp), "percentage points" spelled out once (abstract first-use, then `pp`). Swept abstract/intro/results/conclusion prose.
- **Linker / judge descriptor → named/implicit** — macros retuned (`\entValidator`→"named judge", `\corefValidator`→"implicit judge"); leftover "entity/coreference" prose in `results.tex:119,121,129` + `eval.tex:98` → named/implicit. `rw.tex` keeps "entity linking"/"coreference" as related-work topic.
- **Run-on splits** — `results.tex` summary (was ~80 words), RQ2 + RQ2-answer, worst-component sentence; `motivation.tex:91-97` and `:109-113`. Split into 15–20-word declaratives.
- **Headline framing (Q4)** — abstract + intro both lead with doc-model `+10.0`pp, then doc-code `+5.7`pp file \fone. Abstract reworked; intro `:48` gained the doc-code follow-on.
- **Metric-label mismatch** — `intro.tex:48` "link-level \fone" → "doc-model macro \fone" (matches `results.tex:39` / `conclusion.tex`).
- **JabRef one-liner** — `intro.tex` Para 5 `%TODO` replaced with a one-sentence example (strongest LLM tool ranks first yet never links `preferences`).

## Resolved 2026-06-29 (pass 1)

- **Citation attribution** — added `fuchs_whos_2025` (Artemis) to the cites at `intro.tex:14` and `:48`; the `0.836`/`0.849`/`0.936` numbers are Artemis's.
- **Coverage conflation** — `motivation.tex` reframed to component wording ("a component that a fifth of JabRef's documented sentences describe").
- **Novelty overclaim** — `intro.tex:34` qualified: "the first multi-stage \ac{LLM} workflow … that separates reusable architecture knowledge from the linking step and checks every link against its evidence."
- **"three parts" → "three modules"** — `intro.tex:35` now "three modules" (knowledge + two linkers), judges folded into the linkers.
- **`eval.tex` subsection titles** — `:61/:90/:115` switched to the `\ref{rq:*}` form.
- **`conclusion.tex:12`** — `ArDoCo` → `ARDoCo`.
- **Unused `smelly-discussion.bib`** — `git mv`'d to `archive/`.

---

## Resolved (record)

doc-to-X → `doc-model`/`doc-code` everywhere rendered · wrong `IEEEkeywords` → traceability terms · abstract `validator`/`layer`/`ruler` → `judge`/`module`/`metric` · broken `\autoref{sec:discussion}` → `sec:threats` · rendered `\todo` in `rw.tex` commented out · typo/grammar cluster (`catched`, `machted`, `workfow`, `diffuclut`, "two specific task", "following contributions", …) · `\Artemis`/`\TransArc` macro pass (results + eval) · ARDoCo casing (except conclusion) · judge casing `approach.tex:279-280` → `\entValidator`/`\corefValidator` · "additively to \fone" → FP-removal wording · "operationalize" → "express" · empty `conclusion.tex` written · appendix "ruler" + `SAD-SAM-code` removed.
