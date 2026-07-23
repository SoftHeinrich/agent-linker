# ArchLinker novelty vs Artemis — debate synthesis

Source: 5-agent structured debate (proponent / skeptic / rebuttals / impartial judge),
grounded in the Artemis TAAS25 impl (`baselines/.../connection-generator-ner`) and the
alinker-paper sections (intro, approach). Date: 2026-06-30.

## One-line verdict
ArchLinker's genuine novelty over Artemis is recovering links from **name-free (pronominal)
sentences** via ILinker's cross-sentence antecedent constraint — plus a precision-side
ambiguity map and evidence-typed per-linker judges. Its "runtime knowledge layer" is mostly
externalization hygiene over aliasing Artemis already derives.

## Ranked novelties (what survived cross-examination)

| Strength | Novelty | Delta vs Artemis |
|---|---|---|
| **Strong/genuine** | **ILinker** — implicit-reference linker w/ cross-sentence structural antecedent constraint | Artemis = NER over occurrence lines; entity recovered only if a name *appears*. Name-free pronominal sentence is structurally outside its reach. ILinker binds the pronoun to a component named in an earlier sentence. Both sides converged: difference of *kind*. |
| Moderate | **Ambiguity map** — document-global flag that a component name is an ordinary English word that over-catches | Artemis has no over-catch model; `alternativeNames` only add recall. The map is a precision suppressor that *changes the link set*. |
| Moderate | **Evidence-typed judges** — per-linker judge adjudicates a schema'd evidence bundle (span, reference form, antecedent sentence) | Artemis emits any tier-match at fixed prob 0.92, no evidence object, no check. Novel part = the verification *contract* (e.g. "does the cited antecedent actually name the bound component?"), not the generic act of adding a judge. |
| Moderate | **Size-aware eval suite + long-tail diagnosis** | Metrics are textbook imports; what's new = domain diagnosis + the gap **widens** under it (+10pp link-level → +41pp worst-component, +42pp harmonic). Proves gains land on the tail, not head-shaving. |
| **Weak/incremental** | "Runtime knowledge layer" as a new module | Stripped of the ambiguity map = memoization + separation hygiene. Artemis already derives aliases (primary + `alternativeNames` + occurrences) and grounds vocab via `possibleEntities`. Compute-once changes *when*, not *which links*. This IS Keim's objection. |
| **Weak/incremental** | DLinker split; "first training-free multi-stage workflow" | DLinker = standard prompt decomposition. Training-free / no-labels / no-tuning / multi-stage all already true of Artemis (proponent conceded). |

## Overhyped (drop or reframe)
- "Runtime knowledge layer" as a wholesale new module — only the ambiguity map changes the link set.
- DLinker as a methodological contribution.
- "First training-free, multi-stage LLM workflow" — non-differentiating.
- "We added an evidence-checking judge" — generic generate-then-verify is prior art.
- The size-aware metrics *themselves* — only the diagnosis + widening-gap survive.

## Defensible thesis
The **specific combination**: externalized *ambiguity-aware* knowledge + structurally-novel
implicit-reference linker + evidence-typed per-linker judges. ArchLinker clears the
incrementality bar, but on a narrower base than the current framing advertises.

---

## Draft: tightened "Novelty vs Artemis" paragraph (for rw.tex / intro Para 3)

> The newer LLM line, represented by Artemis, recognizes the architecture entities *named* in
> each sentence and matches them to model components. Because recognition keys on a name that
> appears in the sentence, a sentence that refers to a component only through a pronoun, with no
> name present, falls outside what such a pipeline can recover. ArchLinker targets exactly this
> gap: a dedicated implicit-reference linker resolves the pronoun to an earlier sentence that
> names the component, recovering links from sentences that name nothing. Two further differences
> sharpen precision and reliability: a document-global ambiguity map suppresses component names
> that are ordinary English words and would otherwise over-catch, and a per-linker judge rejects
> any link whose structured evidence does not support it — neither of which the prior line provides.

## Draft: response to Jan Keim's review comments

**Keim: "Artemis/SWATTR also build a knowledge layer used to recover links. How is this different?"**
> Conceded that prior work builds alias/recommendation knowledge. The differentiator is *not* that
> we build knowledge but (i) that the ambiguity map encodes which names over-catch (a
> precision-side signal Artemis lacks; its `alternativeNames` only add recall), and (ii) that
> linking from *name-free* sentences via antecedent resolution is a capability Artemis's
> name-keyed recognition cannot reach. We have reframed the section to lead with these two points
> rather than with externalization/reuse, which we agree is hygiene, not a new mechanism.

**Keim: "Prior work also needs no labelled links or per-project tuning. What is new?"**
> Agreed — we have removed "training-free / no labelled links / no per-project tuning" as novelty
> claims. They describe Artemis too. The contribution is the control structure (knowledge held
> outside the linker + independent evidence-checking judges + the implicit-reference linker), not
> the absence of training.

**Keim: "What exactly is a 'runtime knowledge layer'?"**
> Now stated concretely as two data structures — an alias table and an ambiguity map — computed
> once from the document and component set. We no longer sell the reification itself as the
> novelty; the load-bearing element is the ambiguity map.

## Empirical anchor to keep prominent
Gap widens under size-aware metrics: +10.0pp doc-model link-level → **+41pp worst-component F1,
+42pp harmonic per-component F1** vs Artemis. This is the strongest evidence the gains are not
metric-gaming (head-shaving). Keep it adjacent to any novelty claim.
