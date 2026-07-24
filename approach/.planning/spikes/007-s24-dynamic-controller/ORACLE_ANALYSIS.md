# Oracle analysis: how an ideal controller should prepare the workflow

## What gold revealed

The fixed S21 floors miss 27 gold links across the five projects. The current
bounded alias and anchor candidate generators cover 11 of those misses.
Eligibility counts alone do not predict utility:

| Evidence pool | Gold / eligible | Interpretation |
|---|---:|---|
| alias pool A | 1 / 1 | exact approved alias used as a component reference |
| alias pool B | 0 / 4 | alias token appears only inside an incidental attribute phrase |
| alias pool C | 1 / 13 | one architectural use amid many platform/environment uses |
| anchor pool A | 1 / 1 | unique locally grounded shorthand |
| anchor pool B | 2 / 27 | mostly package/code inventory and generic-name noise |
| anchor pool C | 7 / 11 | repeated locally grounded structural shorthand |

These labels are deliberately anonymized. No project name, component name,
sentence, or benchmark-specific term is transferred into the runtime prompt.

## Ideal preflight profile

Before selecting a phase, the controller needs four views:

1. **Document profile** — scale, linked-sentence coverage, and whether grounded
   examples look like architecture prose, anaphoric prose, package/code
   inventory, captions, or a mixture.
2. **Component profile** — catalog, ordinary-word ambiguity, components with no
   floor links, naming families, and aliases already approved by Phase 1.
3. **Floor profile** — link count, source mix, linked coverage, and underlinked
   components. This says where S21 already invested evidence.
4. **Phase evidence profile** — concrete candidate sentence, referring phrase,
   target, anchor, and grounding basis. Counts are only metadata.

The controller ranks phases by expected novel recall, grounding specificity,
catalog ambiguity risk, and validation cost. It chooses one phase, not a full
up-front list.

## Structured decision pattern

The prompt should request a concise decision record rather than hidden
chain-of-thought:

```json
{
  "assessment": {
    "document_regime": "architecture_prose|mixed|technical_inventory|caption_heavy",
    "catalog_risk": "low|medium|high",
    "best_evidence": "alias_phase4|anchored_reference|none",
    "expected_gain": "low|medium|high",
    "false_positive_risk": "low|medium|high"
  },
  "action": "alias_phase4|anchored_reference|stop",
  "reason": "brief citations to runtime profile fields and candidate evidence"
}
```

This makes decisions auditable without asking for private reasoning or allowing
the controller to propose a link.

## Tool feedback contract

Each phase returns a funnel, not merely a link count:

```json
{
  "tool": "alias_phase4",
  "input": {"eligible": 13, "distinct_targets": 1},
  "validation": {
    "pass1_approved": 2,
    "pass2_approved": 1,
    "consensus_approved": 1,
    "pass_disagreements": 1
  },
  "output": {"new_links": 1, "accepted_targets": ["runtime target"]}
}
```

For anchored recovery the funnel is eligible → resolver-approved →
validator-approved. The next decision uses yield, disagreement, target
concentration, and whether remaining evidence is complementary. A zero-yield
phase exhausts only that evidence channel and does not automatically forbid an
independent phase. Because validators judge candidates individually, a noisy
pool may still be worth invoking when it contains at least one specific,
plausibly component-denoting example. A noisy validation funnel raises the
burden only for another overlapping phase.

## Stop conditions

Stop when the remaining phase has no eligible evidence, its examples are
dominated by technical inventory/caption/generic uses, it overlaps evidence
already exhausted, or its expected marginal value does not justify another
validation pass. Continue only for complementary, project-specific evidence.
