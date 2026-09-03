# approach.tex audit trail

Legacy revision spec (G1-G4 round, 2026-06-28) and the inline `%DONE` code-audit
notes, removed from `sections/approach.tex` on 2026-09-03 by the RW-PASS refactor.
Preserved verbatim: these are the provenance for the numbers in
`approach-stripped-measurements.md` and the reason each retired design was retired.

## Legacy revision spec (removed from the section header)

```tex
% =====================================================================
% REVISION SPEC — global / section-wide directives only.
% Localized fixes stay scattered inline as the %TODO / %HL notes below,
% at the passage they touch (no line numbers here — they drift).
% =====================================================================
% STATUS LEGEND (each G-item carries a >> status line):
%   [DONE]     applied and verified; nothing open. Inspect adjacent prose.
%   [PARTIAL]  some sub-tasks done; the OPEN: line lists what remains.
%   inline marks at the passage: %DONE <ask>-><change>, %FLAG <ask>
%   (needs your decision), %TODO <ask> (localized work still pending).
% Last reconciled against the prose + inline todos: 2026-06-28.
% ---------------------------------------------------------------------
%RW-FIX [header] the OUTSTANDING list below still indexes sec:model-understanding and
%   tab:strategies. Neither exists in this file any more. Drop both entries when this pass lands.
% G1. Terminology & readability sweep (whole section):
%     workflow = framing noun; drop "surface"; "canonical name" ->
%     "verbatim/exact name"; "overshoot" -> simpler; "LLM pass" ->
%     "LLM step"; no "---" (em dash); reuse intro wording, no new claims.
%   >> [DONE] Verified by grep: "surface", "canonical", "overshoot", and
%      "---" occur ONLY in comments, never in live prose. "workflow" is the
%      framing noun throughout; "exact name" and "LLM step" are in use; the
%      validator keeps "pass" as its defined term. Nothing open.
% G2. Three challenges framed as problems OUR workflow solves, with each
%     design choice justified by the insight that removes its challenge.
%   >> [DONE] Restructured to a 1:1 map (option A "Knowledge / Forms /
%      Reliability"): C1 project-knowledge gap -> D1 knowledge module;
%      C2 reference forms -> D2 two linkers; C3 LLM reliability -> D3 judges.
%      Challenges paragraph, nutshell, and the three sec:arch decision leads
%      all rewritten to match; overview figure unchanged.
% G3. Validators as LLM-as-judge (evidence-checkers, not re-classifiers).
%   >> [DONE] "validator" -> "judge" renamed paper-wide in prose (approach,
%      intro, motivation, rw, eval, results) plus abbrev.tex
%      (\entValidator/\corefValidator now render "named/implicit judge").
%      Subsection is now "Evidence-Backed Judges"; contribution renamed to
%      "evidence-backed judging"; "asymmetry" dropped. KEPT (to avoid
%      breaking refs / desyncing figure images): labels sec:validators,
%      rq:validators, fig:rq3-validator and ablation IDs NoValidator/
%      NoEntity/NoCitation.
% G4. Knowledge module: scope down AND de-emphasize.
%   >> [DONE] Over-selling removed (no "failure mode we observed", no load-
%      bearing claim; optional RQ cite left as an inline %TODO at sec:arch).
%      Nutshell no longer leads with the two-table description, and the first
%      sec:arch decision is reframed as the project-knowledge challenge
%      (aliases + ambiguous names), not a name-specific lookup.
% ---------------------------------------------------------------------
% OUTSTANDING LOCALIZED FIXES (stay inline as %TODO; indexed by section):
%   sec:arch              [OPEN] position prior work by mechanism (regex / string
%                         similarity / neural embedding), not the generic glossary/
%                         IR citations -> inline %TODO (overlaps rw.tex). The cite
%                         set itself is fact-checked: 5/5 SUPPORT [DONE].
%   sec:model-understanding  [DONE] swapped in the JabRef 'preferences'
%                         example from the motivation chapter.
%   sec:doc-understanding [OPEN] simplify + cut ~half: drop the "trailing word"
%                         machinery, keep the ambiguity link -> inline %TODO.
%   sec:entity-linker     [DONE] worked example added (S11 -> preferences,
%                         plus the S7 false positive) reusing the motivation.
%   sec:coref-linker      [PARTIAL] illustrative pronoun example + two-step framing
%                         present; OPEN: polish wording + reconcile the alias gate's
%                         home with sec:validators -> inline %TODO.
%   sec:validators        [OPEN] two sub-asks: (a) make the alias gate step 1 of the
%                         judge, (b) add a small prompt-structure sketch -> inline
%                         %TODO. tab:strategies reconcile is moot (table removed).
% =====================================================================
```

## Inline audit notes (removed from the passages)

*carried forward as %RW-CARRY:*
```tex
%RW-CARRY [sec:arch] position prior work by LINKING MECHANISM (regex/lexical, string-similarity,
%   neural-embedding), not the current generic glossary/IR cites. Overlaps rw.tex -- decide at
%   execution whether the mechanism taxonomy lives in sec:arch or related work. Specs-only here.
```

```tex
%DONE fact-check (2026-06-28): all 5 cites SUPPORT their claims (papers fetched + read).
%   furnas1987 = the vocabulary problem (1 referent, many names; ~80-90% single-word-match
%     failure) -> backs the "ordinary English / lexical matching unreliable" line.
%   arora2017 (full PDF) + gemkow2018 (verbatim abstract) = automated glossary-term extraction
%     from requirements -> backs "glossary mining for requirements".
%   howard2013 (software-specific similar-word PAIRS from comment-code) + falleri2010
%     (WordNet-like identifier network: synonymy/hypernymy/hyponymy) -> back "software-side
%     synonym networks". howard outputs pairs, so falleri is the stronger "network" cite;
%     current "in the spirit of ..." phrasing is fine.
%   Residual (low pri): falleri primary abstract anti-bot walled; verdict via HAL record +
%     aggregated snippets. OA PDF at hal-lirmm.ccsd.cnrs.fr/lirmm-00531807/file/icpc.pdf for
%     optional primary-source confirmation.
```

```tex
%DONE 2026-08-10 (code audit): spelling variants are NOT produced by the knowledge module. They are
%   recognized by a deterministic word-signature match inside \linkerB (_spelling_variant_candidates),
%   so the sentence now attributes each part of N(c) to the component that actually produces it.
```

```tex
%DONE 2026-08-15 (rule audit, pilot/rule_audit.py): the four lexical rules the implementation used to
%   carry -- the \linkerB admission filter, the spelling-variant proposer, the as-spelled scan, and the
%   \linkerD proposer -- are one relation at four settings, verified identical on all 3697
%   (name, sentence) pairs of the five benchmarks. The paper now defines the relation once and lets
%   each linker name the cell it scans, instead of motivating four rules separately.
```

```tex
%DONE [order] (2026-09-01, code audit): the "sees only what the earlier ones left unlinked" design and
%   its measured 6.8-FP claim describe machinery removed before this arm: s25 subtracted the settled
%   pairs at every candidate boundary, s79 removed the subtraction, and in the reported arm
%   `_run_linker` passes no link set at all -- duplicates are merged afterwards by `_union`. The
%   structural antecedent constraint went the same way (s80); see sec:coref-linker.
```

```tex
%DONE 2026-08-10 (pilot): the alias scope grade ("global"/"local") is gone -- it only ever filtered
%   the extraction prompt, and dropping it is worth +3.0 TP (p=0.01) at +1.0 FP (p=0.59). The
%   ambiguity map it used to be paired with is gone too (see sec:knowledge), so the "trailing word"
%   machinery and the two-module framing both leave with it.
```

```tex
%DONE [downstream] (2026-08-29) resolved on both counts. (a) No section describes a two-table
%   knowledge module any more: intro.tex, eval.tex and results.tex all say "the knowledge module
%   and the three linker modules", matching "holds one table, the alias table" above. (b) The
%   no-knowledge ablation was re-measured without the ambiguity map, on this arm
%   (2026-09-02: s_linker110_noknow, three runs per backend -- rebased off s_linker92a_noknow
%   when s110 became the reported arm); results.tex reads 7.6pp \fone / 11.9pp \ftwo.
%   See the %DONE [noknow] block in results.tex.
```

```tex
%DONE 2026-08-10 (code audit): dropped "and a short reason" -- the bundle's extraction_rationale field is
%   the same constant string for every candidate, so it carries no per-candidate information to report.
```

```tex
%DONE 2026-08-10 (pilot): an earlier revision sampled this prompt twice and unioned the two candidate
%   sets. Five runs per side on all five projects: TP -1.2 (p=0.30), FP -1.2 (p=0.42) -- neutral, so
%   \approach samples once and the paper no longer has to explain a self-consistency guard.
%   (2026-09-01: moot -- the extraction prompt this note is about is gone; the proposer is a scan.)
```

```tex
%DONE [entity prompt] (2026-09-01, code audit): the {entity} "Named-Reference Extraction" prompt box
%   was removed with the extractor it documented. In the reported arm the \linkerB proposer issues no
%   prompt at all (`Extracted: N (scan, 0 calls)` in every recorded run log), so a prompt sketch here
%   would document a call the workflow does not make.
```

```tex
%DONE [nesting] (2026-09-02, arm rebase; standalone 2026-09-02) the refusal arrives with
%   s_linker109 and is INLINED in the reported s_linker110, which is now a standalone file with
%   no linker base class: it is HEAD DELTA 2 there, `_only_inside_another_name` filtering
%   `_scan`, block-identical to s_linker109's and checked as such by
%   pilot/test_s110_shortlist.py. Numbers from pilot/test_s109_nesting.py (345 checks, 12
%   recorded runs, no LLM calls). The catalog-only
%   restriction is load-bearing: the alias form of the same predicate costs 3 gold links in one
%   recorded luna run. The Redis PubSub / Redis DB and HTML5 Server / HTML5 Client cases above
%   were replayed against the predicate itself, not paraphrased from the docstring.
```

```tex
%NOTE: the simplified motivation figure has no pronoun case, so this one sentence is illustrative, not a benchmark quote -- swap in a real coreference example if one is available.
```

```tex
%DONE [shortlist] (2026-09-02, arm rebase; standalone 2026-09-02) this IS s_linker110, the reported
%   arm: `_prompt_coref` and `_named_before` are the only two methods it adds over the workflow it
%   carries (HEAD DELTA 3 -- the file is standalone now, so they are declared rather than overridden),
%   and `_named_before` builds the list with `_states_a_name`, the same whole-name row of the relation
%   sec:arch defines. Stage numbers: pilot/reading_pilots.py --arms
%   control shortlist2, 3 samples x 5 projects per model. Shortlist size and the median-2 antecedent
%   distance: pilot/test_s110_shortlist.py, 224 checks, no calls. E2E (three paired runs a model,
%   pilot/run_consolidation_e2e.sh) is what results.tex reports; do not restate it here.
```

```tex
%DONE [sec:coref-linker] (2026-09-01, code audit) the "pick ONE home for the alias gate" question is
%   moot: the gate is not in either home any more. s80 removed it after measuring it at TP +/-0.0 and
%   FP +/-0.0 on the pairs \linkerC alone contributes, and `_resolve_references` in the reported arm
%   checks only that a resolution reports an antecedent and that both sentence numbers exist.
```

```tex
%DONE 2026-08-10 (code audit): "a hallucinated link is caught by a mismatch between bundle and sentence"
%   held only for \linkerD, whose judge verifies claim-substring + anchor membership in code
%   (_classify_denotations). The \linkerB and \linkerC judges request a claim quote and discard it
%   (_run_validation_pass reads only "approve"). Scope now stated explicitly. Updated 2026-09-01: the
%   p1/p2 split this note referred to is gone, and _review_identity_batch with it.
```

```tex
%DONE [sec:validators] (2026-09-01, code audit) both open items resolved against the reported arm.
%   (a) the coref alias-gate has no home to pick: s80 removed it (see sec:coref-linker). (b) the
%   p1/p2 two-pass \entValidator was retired before this arm -- `_validate_with_evidence` issues one
%   `_run_validation_pass` per batch with an empty focus line, and the 1-pass call model reproduces the
%   recorded per-project LLM call counts exactly on all five projects (9/31/10/19/6 for terra run 1),
%   where a 2-pass model predicts 11/35/11/21/7. All three judges rule once.
```



## All remaining inline annotations, cleared 2026-09-03

Everything below was removed from the section body so that the new N1-N10 spec header
is the section's only spec. It includes the RW-PASS working marks from this session's
planning rounds, the old numbered section dividers (whose contents were stale: they
still described a two-linker shape), and several commented-out alternative sentences.

```tex
% =====================================================================
% RW-PASS 2026-09-03 -- reframe to the evidence spine (evidence, hosting
% reference form as its first move). This is an EDIT pass over the existing
% flow, not a rewrite: paragraph order and subsection skeleton stay.
% Marks from this pass are tagged %RW so they grep apart from the older
% %TODO / %DONE / %FLAG notes above:
%   %RW-EDIT <what>  adjust this passage in place, keep its role in the flow
%   %RW-CUT  <what>  remove: measured justification or retired alternative
%   %RW-FIX  <what>  factual error against the reported arm
%   %RW-ADD  <what>  new material needed at this point
%   %RW-TERM <what>  evidence-backed / context-augmented -> evidence-augmented
% Decisions this pass implements:
%   - ONE root cause, not three independent challenges (intro.tex lock).
%   - NO measured numbers and no p-values anywhere in this section; every
%     design choice is justified conceptually. The numbers live in results.
%   - Aliases are a minor supporting subsection, not a contribution.
%   - Evidence checking is the centre of gravity.
%   - Paragraphs run 3-4 compact sentences, one claim per sentence.
%   - No length target; compression comes from dropping the measurements.
% STRUCTURE (governs EVERY design passage in this section, not just the opener):
%     PHENOMENON -> CHALLENGE -> INSIGHT -> HOW THE INSIGHT GUIDES THE DESIGN.
%   Per the argument playbook's signature design paragraph (T6 / Template 2): walk the reader
%   through what is observably true, name what that makes hard, land an INTERPRETATION sentence
%   that plants the exploitable property, and only then say what the design does about it. A
%   passage that opens on mechanism has skipped three moves. Each subsection repeats the shape at
%   its own scale: the form it handles (phenomenon), what that form fails to settle (challenge),
%   what can be settled exactly anyway (insight), the proposer/judge pair it licenses (design).
%   Do NOT re-evidence a phenomenon sec:motivation has already established; point and move on.
% Bridge sentences the section must land:
%   "The form of the reference is what evidence there is."
%   "A judge can be no stricter than the evidence its linker reports."
% =====================================================================
%RW-EDIT [lead] keep the scope sentence and the input/output definition verbatim in role,
%   but move "training-free" out of the thesis position: the story demotes it to a property of the
%   interface, not a differentiator (intro spec item P6). Say what the workflow does first.
%RW-EDIT [challenges] ADJUSTED IN PLACE 2026-09-03 (rev 5), not replaced. What changed and why:
%   (a) topic sentence: "three challenges, one design decision each" -> ONE root cause. The intro
%       locks a single problem with one implication ("ONE problem + its implication. NOT three
%       separate problems"), and three co-equal challenges is what made the section read as three
%       competing contributions. Rev 3 states the actual cause instead of naming its two effects.
%       Rev 1 was abstract ("a sentence rarely gives the whole of what a link needs"); rev 2 was a
%       conjunction of the two observations ("several ways to point, each leaving different
%       evidence"). Neither was a reason. The cause is in this section's OWN definition of a link,
%       three lines above: a link asserts that a sentence DISCUSSES a component, and whether it
%       does is not settled by which words the sentence contains. One cause, two symptoms in
%       OPPOSITE directions -- words missing where the discussion is real, words present where it
%       is not -- which is the "ONE problem + its implication" shape intro.tex locks. Both symptoms
%       are already in fig:example: a refer-back sentence, and S7's "preferences".
%   (e) rev 4 puts the passage into the section's mandated shape. Rev 3 had the right cause but
%       delivered it as cause -> effect -> effect -> thesis, which states a problem and a response
%       and skips the two moves that earn them. Now: PHENOMENON (documentation explains a system,
%       it does not identify parts) -> CHALLENGE (so word presence does not settle discussion, and
%       matching fails in both directions) -> INSIGHT (the words are still all anyone can point at,
%       and how much of a name a sentence gives IS exactly settleable even when its meaning is not)
%       -> DESIGN (split along that form; a proposer and a judge per form). Two paragraphs of four
%       sentences: phenomenon+challenge, then insight+design.
%   (f) the three-form list moves into the INSIGHT move, where it belongs: it is no longer a list of
%       difficulties but the enumeration of what can be settled exactly. Same wording, new job.
%   NOTE the ordinary-English word appears here as the PROBLEM. That is not a walk-back on dropping
%       the ambiguity framing from the alias table: the table does not resolve it, the judge that
%       reads the sentence does. That is exactly what this paragraph sets up.
%   (g) rev 5 makes the PHENOMENON tangible. Rev 4 had the right four moves but opened on an
%       abstract proposition ("whether a sentence discusses a component is not settled by the words
%       it contains"), which is epistemology, not something an SE reviewer can see. The corpus rule
%       is the opposite order: "concrete motivating example BEFORE formalism" / "worked micro-example
%       before formalisation". So the passage now opens on a real sentence from the documentation and
%       lets the general claim fall out of it in one clause. Sentences shortened to the ~16-word mean
%       the playbook reports for approach sections; the abstract version averaged twice that.
%   (h) the example is POINTED AT, not re-argued: sec:motivation already introduces this sentence, so
%       the opener quotes it and moves on rather than re-establishing the case.
%   NOTE the phenomenon claim ("documentation explains a system rather than identifying its parts")
%       is asserted, not measured, on purpose: sec:motivation establishes it, and this section must
%       not re-argue it (and cannot, with no numbers allowed here).
%RW-FIX [quoted sentence] the quoted sentence is REAL and verbatim from the benchmark document
%   (benchmark/jabref/text_2021/jabref.txt, sentence 7), and its gold standard has exactly one link,
%   to gui. Keep it verbatim when fig:example is renumbered, and do NOT claim the same document also
%   contains a name-free refer-back case: it does not. The "other sentences" claim below is scoped to
%   architecture documentation in general on purpose, and other benchmark projects carry those cases.
%   (b) the vocabulary item MOVED OUT of this paragraph, to the head of sec:knowledge, where the
%       aliases are actually described. It was the only item not about the reference form, and
%       holding it here separated the two halves that belong together.
%   (c) the two symptoms are adjacent, missing-words first and present-words second, because they
%       are what the three linkers and the judges respectively answer. That adjacency IS the
%       argument: one cause, so one workflow, not three features.
%   (d) the third item stops being a challenge and becomes the thesis the rest of the section argues.
%   The First / Second / Third scaffolding goes with the reorder: two connected moves, not a list.
%RW-EDIT [fig caption] reword to the evidence framing: the form of the reference decides which
%   linker proposes and how strict its judge can be. Drop "ordered by" -- the linkers are independent.
%RW-FIX [fig Description] "a coreference linker for sentences giving no name" is wrong. In the
%   reported arm EVERY sentence goes to the resolver in context; there is no pronoun or name-free
%   filter. Also "three linkers run in sequence" overstates: they are independent and merge by pair.
%RW-EDIT [nutshell] "answers the three challenges" -> the three steps of one decision. Keep the
%   (i)/(ii)/(iii) rhythm and the <=200-word budget; keep the merge sentence.
% Because the knowledge is built once, \approach resolves each alias the same way for every sentence instead of re-deciding it on every call.
% -----------------------------------------------------------------
% 2.1 Architecture: the knowledge-layer decision + the
% reference-form decision jointly justify the 2-layer + 2-linker shape.
% -----------------------------------------------------------------
%RW-EDIT [sec:arch lead] "Three decisions fix the shape" -> the evidence definition, which is what
%   the rest of the subsection needs: to decide a link on its evidence, the workflow must first say
%   what the evidence is. Pin the word once here -- evidence = words in the document that can be
%   pointed at -- and never use it loosely afterwards.
%RW-CUT [decision 1] the whole "gathers it once, up front, rather than leaving each linker to work
%   it out on its own" framing goes. It is the retired novelty axis: the baseline also builds a
%   reusable alias record once per document, so "computed once" and "knowledge kept separate from the
%   linker" separate us from nothing. Verified against the released baseline source.
%RW-EDIT [decision 1] what survives moves to sec:knowledge and shrinks: the document coins names the
%   model does not carry, so they have to be found before any form can be recognized. The five
%   vocabulary/glossary citations move with it.
%RW-CARRY [sec:arch] position prior work by LINKING MECHANISM (regex/lexical, string-similarity,
%   neural-embedding), not the current generic glossary/IR cites. Overlaps rw.tex -- decide at
%   execution whether the mechanism taxonomy lives in sec:arch or related work. Specs-only here.
% , so a linker decides each link over a candidate set the module has already narrowed instead of discovering names and judging the link in one step.
%RW-EDIT [decision 2] this becomes the FIRST move of the subsection, not the second: the form of the
%   reference is what evidence there is, so one linker and one judge per form. Keep N(c).
%RW-EDIT [relation] collapse the formalism. The implementation has exactly TWO forms (whole name at
%   any case; one word of a name under inflection). The three-valued Fidelity axis indexes one live
%   value, so the two-dimensional relation is machinery the reader pays for and never uses. Keep a
%   light statement of the two forms and drop "as spelled" / "any spelling" from the definition.
%RW-EDIT [tab:forms discussion] rewrite with NO numbers. The point is conceptual: the looser the
%   form, the weaker the evidence, and the strictness of the judge behind it is set to match. Keep
%   "nothing in this deterministic layer admits a link -- it can only end a case" and both places it
%   does so; that sentence is load-bearing for the name-matching objection below.
%RW-EDIT [tab:forms] replace the measurement table with a numbers-free DESIGN table: form ->
%   which linker scans it -> how strict that linker's judge is (approve by default / target-blind /
%   reject when uncertain). The reach and precision figures move to the results section.
%RW-CUT [nesting of forms] this paragraph goes to threats to validity. It exists to support the two
%   forms the workflow does NOT scan, and its conclusion is a reach limitation (a component the
%   document only ever writes split is out of reach), which belongs with the other limitations.
%RW-EDIT [three linkers] keep this paragraph and its payoff; tighten to 3-4 sentences. Keep
%   "a property of the sentence, not of the link set" and the closing sentence, which is one of the
%   two bridge sentences: a judge can be no stricter than the evidence its own linker reports.
%RW-ADD [asymmetry] ADD one short paragraph after this one, and it is the paragraph that answers the
%   objection the section currently leaves open. Recognizing a form is a matter of fact and is settled
%   in code; whether the sentence makes an architectural claim about the component is not, and only a
%   judge decides that. So \approach matches names MORE freely than prior work, not less, because
%   matching no longer decides anything. State it without numbers, and claim no guarantee: this is a
%   design discipline, not soundness.
%RW-EDIT [decision 3] keep as the hand-off into sec:validators, but re-lead: a judge shown only a
%   candidate link has nothing to check it against. Drop "the reliability challenge" wording, since
%   there is no longer a list of three challenges to be the third of.
% A judge sits after each linker, and the number of checking passes matches how hard the decision is (\autoref{sec:validators}).
% Together these decisions give the three-stage workflow of \autoref{fig:approach-overview}: knowledge discovery, the two reference-form linkers, and LLM based judges.
% -----------------------------------------------------------------
% 2.3 Knowledge discovery. The two modules address the
% ordinary-English ambiguity and the alias-discovery problem.
% -----------------------------------------------------------------
%RW-ADD [sec:knowledge lead] the vocabulary sentence moved out of the challenges paragraph lands
%   here, as this subsection's opening challenge->solution couplet. Verbatim, ready to drop in:
%     The project's vocabulary is not given in advance: a component appears under aliases the
%     document coins as it goes, so what counts as a name must be found in the document before any
%     form can be recognized.
%RW-EDIT [sec:knowledge] retitle to "Alias Discovery" and flatten the \subsubsection into it. The
%   label stays sec:knowledge so nothing downstream breaks. Rationale: this is now a short supporting
%   subsection, and a grand heading over two sentences advertises importance the content does not have.
%   Target 3-4 sentences: the document coins short forms and alternative names; a confirmation step
%   keeps the ones the document establishes; a term two components both claim names neither and is
%   dropped; a confirmed alias is then a name of its component like any other.
%RW-EDIT [sec:doc-understanding] fold into the parent subsection; keep the label. Keep the two alias
%   kinds and the confirming judge; drop "so this is the hard kind", which sells the module up.
%RW-EDIT [alias example] use the running example: the document ties "command line interface" to the
%   cli component. That is a real explicit renaming in the benchmark document and it is also the name
%   that makes the partial-name case below possible.
%RW-CUT [prompt box] comment this box out for now, do not delete. Keep a one-clause description of
%   its shape in the prose so it can be restored without rewriting.
% -----------------------------------------------------------------
% 2.4 LinkerB: the four named forms collapse into one extraction
% step once the knowledge layer iss available.
% -----------------------------------------------------------------
% Without the knowledge module, a named-mention linker would handle the all named mention forms from scratch.
% The alias table removes this need: once it exists, every named form is just a sentence in $s$ that matches the exact name or one of the recorded aliases, so a single extraction step covers them all.
%RW-ADD [why a scan] replace with the conceptual reason, which is stronger anyway and needs no
%   numbers: a scan reaches every sentence, whereas a model asked to list the mentions can only ever
%   report a subset of them, and a mention that is never proposed is one no judge can recover.
%RW-EDIT [worked example] renumber to the adapted figure's S-numbers. This is the best paragraph in
%   the subsection -- a real name match and a real false positive side by side -- so keep its shape and
%   only change the references and the sentence quoted.
% -----------------------------------------------------------------
% 2.5 LinkerD: one word of a name is weak evidence, so the judge
% splits: denotation without the target, then grounded identity.
% -----------------------------------------------------------------
%RW-EDIT [partial-name example] use the running example instead: the alias "command line interface"
%   is a multi-word NAME, so a sentence writing only "the command line" carries one word of it. Those
%   words are ordinary English, which is exactly why this judge is shown no target. Keep the second
%   project's case only if the nesting refusal below still needs it.
%RW-CUT [prompt box] comment out for now, do not delete. Of the four boxes this is the one whose
%   SHAPE is the argument (the target is withheld), so it is first in line to be restored.
% -----------------------------------------------------------------
% 2.6 LinkerC: the structural antecedent check removes
% alias-failing candidates without LLM judgment.
% -----------------------------------------------------------------
%RW-FIX [no pronoun filter] FACTUALLY WRONG against the reported arm. There is no pronoun regex and
%   no trigger list: every sentence in the document goes to the resolver in context, batched. A
%   narrowing filter was proposed and refused twice. Rewrite so the sentence says that -- it is also
%   a design claim (no hand-written trigger list) and a cost claim, not just a correction.
%RW-EDIT [coref example] replace the invented example with the adapted figure's refer-back sentence,
%   whose antecedent is the sentence that introduced the cli component. Delete the %NOTE below: it
%   says the figure has no pronoun case, which was true of the benchmark document but is no longer
%   true of the adapted figure.
%RW-CUT [prompt box] comment out for now, do not delete.
% -----------------------------------------------------------------
% 2.6 Validators: evidence over re-classification + epistemic
% asymmetry jointly justify asymmetric pass counts.
% -----------------------------------------------------------------
%RW-TERM [title] retitle to "Evidence-Augmented Judges". The judges are named for what their INPUT
%   carries, which is exactly true of all three; "backed" asserts something about the link, and the
%   full-name judge approves by default with its quote unverified, so "backed" over-claims. The macro
%   in abbrev.tex changes once and every call site follows.
%RW-EDIT [judges lead] re-lead the subsection with the on-ramp, then sharpen: a judge shown only a
%   candidate link has nothing to check it against, so each judge is shown the words the linker
%   matched and the sentences around them -- and that context IS the evidence the link rests on. Keep
%   the two-observation frame; it still works.
%RW-TERM [context-augmented] this sentence already says "we augment the judge's input with that
%   evidence bundle" and then names the result "context-augmented". Rename to evidence-augmented and
%   the sentence becomes self-consistent. "context-augmented" is retired paper-wide (also intro.tex).
%RW-CUT [contribution claim] "We treat it as a method contribution" goes. The intro lists three
%   contributions and this is not one of them, and contributions are stated once and never recapped.
%   Keep the naming sentence: the design is made by explaining the mechanism, not by asserting a label.
%RW-TERM rename \evidenceBacked judging -> evidence-augmented judging.
%RW-CUT [prompt box] comment out for now, do not delete. This is the second box in line to be
%   restored: it shows one template carrying two different rubrics.
%RW-TERM box title -> "Evidence-Augmented Judge".
%RW-ADD [cost closer] add a short paragraph here and leave it COMMENTED OUT for now. Written
%   structurally, with no measured call counts: the workflow spends a small fixed number of model
%   decisions per document, judging is batched rather than asked per pair, and there is no controller
%   and no retry loop. This is the "overhead / cost" closer the style guide asks a design section to
%   end on, and it is the natural home for the batch and window sizes.
```
