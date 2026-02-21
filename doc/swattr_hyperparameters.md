# SWATTR Hyperparameter Reference

Comprehensive documentation of all tunable parameters in the ARDoCo SWATTR (SAD-SAM) Java implementation that affect precision and recall of traceability link recovery.

## Pipeline Overview

SWATTR processes documents through three sequential stages, each containing multiple **Informants** (heuristic processors) that vote on candidates with configurable confidence scores:

```
Text Extraction → Recommendation Generator → Connection Generator
  (NounMappings)    (RecommendedInstances)     (TraceLinks)
```

Each informant contributes a probability/confidence score. These are aggregated (default: arithmetic mean) to produce final link probabilities. Higher aggregate probability = higher confidence in a trace link.

---

## Stage 1: Text Extraction

Extracts noun/phrase mappings from documentation text. Each informant identifies candidate words and assigns them NAME and/or TYPE mappings with a confidence score.

### CompoundAgentInformant

Extracts compound words (multi-part constructions) and special named entities (CamelCase, snake_case).

| Parameter | Type | Default | @Configurable |
|-----------|------|---------|---------------|
| `compoundConfidence` | double | **0.6** | Yes |
| `specialNamedEntityConfidence` | double | **0.6** | Yes |

**Behavior:**
- Finds compound words via `CommonUtilities.getCompoundWords()` and adds them as NAME mappings with `compoundConfidence`
- Detects CamelCase (`isCamelCasedWord`) and snake_case (`nameIsSnakeCased`) words, adds as NAME mappings with `specialNamedEntityConfidence`

**Tuning effect:**
- Raise → more compound/structured names become candidates (recall up, precision may drop)
- Lower → only high-evidence compounds survive aggregation

### NounInformant

Baseline noun extraction based on POS tags. Lowest default confidence of all informants.

| Parameter | Type | Default | @Configurable |
|-----------|------|---------|---------------|
| `probability` | double | **0.2** | Yes |
| `nameOrTypeWeight` | double | **0.5** | Yes |

**Behavior:**
- Filters words: length > 1, starts with letter
- POS tags NOUN_PROPER_SINGULAR, NOUN_PROPER_PLURAL, NOUN → NAME + TYPE with `probability * nameOrTypeWeight`
- POS tag NOUN_PLURAL → TYPE only with `probability`

**Tuning effect:**
- `probability` is the **most conservative** default (0.2) — raising it significantly increases extraction volume
- `nameOrTypeWeight` balances NAME vs TYPE contribution (0.5 = equal)

### InDepArcsInformant

Uses Stanford CoreNLP incoming dependency arcs. Highest default confidence (1.0) reflecting strong syntactic evidence.

| Parameter | Type | Default | @Configurable |
|-----------|------|---------|---------------|
| `probability` | double | **1.0** | Yes |
| `nameOrTypeWeight` | double | **0.5** | Yes |

**Behavior:**
- APPOS, NSUBJ, POSS dependencies → NAME + TYPE with `probability * nameOrTypeWeight`
- OBJ, IOBJ, POBJ, NMOD, NSUBJPASS → NAME + TYPE with `probability * nameOrTypeWeight`
  - If word has indirect determiner (a/an) as pre-word → also adds TYPE with full `probability`

**Tuning effect:**
- Already at maximum (1.0). Lowering reduces influence of dependency-based extraction.
- This is the strongest extraction signal — modify with caution.

### OutDepArcsInformant

Uses outgoing dependency arcs. Slightly lower default than InDepArcs.

| Parameter | Type | Default | @Configurable |
|-----------|------|---------|---------------|
| `probability` | double | **0.8** | Yes |
| `nameOrTypeWeight` | double | **0.5** | Yes |

**Behavior:**
- AGENT, RCMOD → NAME + TYPE with `probability * nameOrTypeWeight`
- NUM, PREDET → TYPE only with full `probability`

**Tuning effect:**
- Default 0.8 (vs 1.0 for InDepArcs) reflects weaker evidence from outgoing edges
- Useful for relative clauses and agent relations

### SeparatedNamesInformant

Extracts words containing separators (dots, colons, dashes, underscores) — typically package-qualified or hyphenated names.

| Parameter | Type | Default | @Configurable |
|-----------|------|---------|---------------|
| `probability` | double | **0.8** | Yes |

**Behavior:**
- Checks `CommonUtilities.containsSeparator()` for dots, colons, dashes, underscores
- Excludes FOREIGN_WORD POS tags
- Adds as NAME mapping with `probability`

**Tuning effect:**
- Raise → package-qualified names like `auth.service` get higher weight
- Lower → reduces extraction of potentially ambiguous separated constructs

### MappingCombinerInformant

Merges similar phrase mappings based on cosine similarity. Unlike others, this is a **threshold** (not a confidence score).

| Parameter | Type | Default | @Configurable |
|-----------|------|---------|---------------|
| `minCosineSimilarity` | double | **0.4** | Yes |

**Behavior:**
- Finds phrase mappings with cosine similarity > threshold (using `PhraseMappingAggregatorStrategy.MAX_SIMILARITY`)
- Merges if underlying noun mappings are equally sized and each can be paired with a similar counterpart

**Tuning effect:**
- Raise (e.g., 0.6) → only very similar phrases merge (more distinct recommendations, potentially duplicates)
- Lower (e.g., 0.3) → more aggressive merging (fewer but broader recommendations)

---

## Stage 2: Recommendation Generator

Creates model element recommendations from text extraction results. Each recommended instance aggregates confidence from contributing informants.

### NameTypeInformant

Matches adjacent NAME-TYPE patterns against model element types. Highest confidence.

| Parameter | Type | Default | @Configurable |
|-----------|------|---------|---------------|
| `probability` | double | **1.0** | Yes |

**Behavior (four patterns using `getSimilarTypes()`):**
1. **Name BEFORE Type:** Word is similar to model type, pre-word has NAME mapping → create recommendation
2. **Name AFTER Type:** Word is similar to model type, next-word has NAME mapping → create recommendation
3. **Name-or-Type BEFORE Type:** Word is model type, pre-word has NAME or TYPE mapping → create recommendation
4. **Name-or-Type AFTER Type:** Word is model type, next-word has NAME or TYPE mapping → create recommendation

**Tuning effect:**
- Already at 1.0. The four pattern variants ensure broad coverage of name-type adjacency.

### CompoundRecommendationInformant

Creates recommendations from compound nouns and special named entities discovered in Stage 1.

| Parameter | Type | Default | @Configurable |
|-----------|------|---------|---------------|
| `confidence` | double | **0.8** | Yes |

**Behavior:**
1. Finds compound noun mappings (via `isCompound()`), extracts type mappings, creates recommendations
2. Checks adjacent words for noun-tagged type mappings → creates compound recommendations
3. Creates recommendations for CamelCase/snake_case words from NAME mappings

**Tuning effect:**
- Raise → compound-based recommendations become stronger candidates
- Lower → conservative compound extraction

### ProjectNameInformant

Penalizes recommendations that contain the project name. Acts as a **precision filter**.

| Parameter | Type | Default | @Configurable |
|-----------|------|---------|---------------|
| `penalty` | double | **Double.NEGATIVE_INFINITY** | Yes |

**Behavior:**
- Gets project name from pipeline data
- Expands each word left/right to match project name segments
- If match found → applies `penalty` to instance probability

**Tuning effect:**
- Default `-∞` completely eliminates project-name candidates
- Raise toward 0 → less aggressive filtering (allows some project-name matches through)

### Probability Aggregation (RecommendedInstanceImpl)

Each `RecommendedInstance` aggregates probabilities from all contributing informants:

```
mappingProbability = RMS(max(nameMappings.P(NAME)), max(typeMappings.P(TYPE)))
ownProbability     = AVERAGE(all informant confidences)
finalProbability   = AVERAGE(mappingProbability, ownProbability)
```

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| Global aggregator | `RecommendedInstanceImpl` | `AVERAGE` | How informant votes combine |
| `WEIGHT_INTERNAL_CONFIDENCE` | `RecommendedInstanceImpl` | **0** | Balance: 0=equal, +2=internal-heavy, -2=mapping-heavy |

**Available aggregation functions** (in `AggregationFunctions` enum):

| Function | Formula | Character |
|----------|---------|-----------|
| AVERAGE | Sum / Count | Balanced |
| MEDIAN | Middle value | Outlier-robust |
| HARMONIC | Count / Sum(1/x) | Penalizes low values |
| ROOTMEANSQUARED | sqrt(Sum(x^2)/n) | Emphasizes high values |
| CUBICMEAN | cbrt(Sum(x^3)/n) | Strongly emphasizes high values |
| MAX | Maximum | Optimistic (any evidence wins) |
| MIN | Minimum | Pessimistic (all must agree) |

---

## Stage 3: Connection Generator

Creates final trace links between recommended instances and model elements.

### ExtractionDependentOccurrenceInformant

Direct lexical matching of text words against model endpoint names/types.

| Parameter | Type | Default | @Configurable |
|-----------|------|---------|---------------|
| `probability` | double | **1.0** | Yes |

**Behavior:**
- For each word, checks similarity to model endpoint names (via `isWordSimilarToEntity()`)
- Skips words that don't start with capital letter AND are not NN* POS-tagged
- Adds as NAME mapping with `probability`; also checks type similarity

### InstantConnectionInformant

Creates connections between recommended instances and model endpoints.

| Parameter | Type | Default | @Configurable |
|-----------|------|---------|---------------|
| `probability` | double | **1.0** | Yes |
| `probabilityWithoutType` | double | **0.8** | Yes |

**Behavior:**
1. Finds most similar recommended instances to each model endpoint by reference
2. Uses `probability` if instance has type mappings, `probabilityWithoutType` otherwise
3. Also creates links for instances equal/similar to model endpoints

**Tuning effect:**
- The gap between 1.0 and 0.8 encodes how much type information matters
- Raise `probabilityWithoutType` → trust name-only matches more (recall up)
- Lower `probabilityWithoutType` → require type evidence for high confidence (precision up)

### NameTypeConnectionInformant

Pattern-based connections using adjacent NAME-TYPE word pairs (connection-level analog of NameTypeInformant).

| Parameter | Type | Default | @Configurable |
|-----------|------|---------|---------------|
| `probability` | double | **1.0** | Yes |

**Behavior:**
Same four patterns as NameTypeInformant but at connection level, using `tryToIdentify()` to match words to specific model entities.

### ReferenceInformant

Creates connections based on NAME mappings similar to model instance names.

| Parameter | Type | Default | @Configurable |
|-----------|------|---------|---------------|
| `probability` | double | **0.75** | Yes |

**Behavior:**
- For each model endpoint, finds noun mappings (NAME kind) similar to endpoint name
- Creates recommendations with `probability`

**Tuning effect:**
- Most conservative connection informant (0.75)
- Raise → more reference-based links (recall up)
- Lower → stricter reference matching (precision up)

### NerConnectionInformant (Embedding-based, newer addition)

Uses Named Entity Recognition + embedding similarity for matching. **Not @Configurable** — uses hardcoded constants.

| Parameter | Type | Default | @Configurable |
|-----------|------|---------|---------------|
| `DEFAULT_PROBABILITY` | double | **0.92** | No (static final) |
| `EMBEDDING_SIMILARITY_THRESHOLD` | double | **0.6** | No (static final) |

**Behavior (three-phase matching, `@Deterministic`):**
1. **Direct similarity:** `SimilarityUtils.areWordsSimilar()` on names and alternative names
2. **Weak similarity:** `SimilarityUtils.areWordsOfListsSimilar()` on CamelCase-split name parts
3. **Embedding similarity:** OpenAI `text-embedding-3-large` cosine similarity (requires `OPENAI_API_KEY`)
   - Only fires if phases 1-2 don't match
   - Threshold: cosine similarity >= 0.6

**Tuning effect (requires code change):**
- Lower `EMBEDDING_SIMILARITY_THRESHOLD` (0.6 → 0.5) → more semantic matches (recall up, may add false positives)
- Raise (0.6 → 0.7) → stricter semantic matching (precision up)

---

## Global Word Similarity Configuration

**File:** `core/framework/common/src/main/resources/configs/CommonTextToolsConfig.properties`

These thresholds affect **all word matching** across the entire pipeline. They are the single most impactful parameters.

### Jaro-Winkler Similarity

| Parameter | Default | Effect |
|-----------|---------|--------|
| `jaroWinkler_Enabled` | **true** | Enable/disable |
| `jaroWinkler_SimilarityThreshold` | **0.90** | Minimum similarity for two words to be considered matching |

Jaro-Winkler is prefix-aware: words sharing a common prefix get a bonus. At 0.90, this is strict — only near-identical words match.

### Levenshtein Distance

| Parameter | Default | Effect |
|-----------|---------|--------|
| `levenshtein_Enabled` | **true** | Enable/disable |
| `levenshtein_MinLength` | **2** | Minimum word length for edit-distance matching |
| `levenshtein_MaxDistance` | **1** | Maximum allowed edit operations |
| `levenshtein_Threshold` | **0.90** | Dynamic threshold multiplied by shortest word length |

At MaxDistance=1, only single-character differences are tolerated (typos, plurals like "Service" vs "Services").

### Recommendation Grouping

| Parameter | Default | Effect |
|-----------|---------|--------|
| `getMostRecommendedIByRef_MinProportion` | **0.5** | Minimum fraction of name parts that must match for instance selection |
| `getMostRecommendedIByRef_Increase` | **0.05** | Step size for progressively expanding match clusters |

### Word Separators

| Parameter | Default |
|-----------|---------|
| `separators_ToContain` | `. :: : _` |
| `separators_ToSplit` | `\. :: : _` |

---

## Summary: All Parameters Ranked by Impact

### Tier 1 — Global, Highest Impact

| Parameter | Default | Location | Change to increase recall | Change to increase precision |
|-----------|---------|----------|--------------------------|------------------------------|
| `jaroWinkler_SimilarityThreshold` | 0.90 | CommonTextToolsConfig | Lower to 0.85 | Raise to 0.92+ |
| `getMostRecommendedIByRef_MinProportion` | 0.5 | CommonTextToolsConfig | Lower to 0.4 | Raise to 0.6 |
| `levenshtein_MaxDistance` | 1 | CommonTextToolsConfig | Raise to 2 | Keep at 1 |
| Global aggregator | AVERAGE | RecommendedInstanceImpl | Switch to MAX | Switch to HARMONIC |

### Tier 2 — Stage-Level, Moderate Impact

| Parameter | Default | Informant | Change to increase recall | Change to increase precision |
|-----------|---------|-----------|--------------------------|------------------------------|
| `probability` (Noun) | 0.2 | NounInformant | Raise to 0.4+ | Lower to 0.1 |
| `probabilityWithoutType` | 0.8 | InstantConnectionInformant | Raise to 0.9+ | Lower to 0.6 |
| `probability` (Reference) | 0.75 | ReferenceInformant | Raise to 0.9 | Lower to 0.5 |
| `EMBEDDING_SIMILARITY_THRESHOLD` | 0.6 | NerConnectionInformant | Lower to 0.5 | Raise to 0.7 |
| `minCosineSimilarity` | 0.4 | MappingCombinerInformant | Lower to 0.3 | Raise to 0.6 |
| `penalty` | -inf | ProjectNameInformant | Raise toward -5 | Keep at -inf |

### Tier 3 — Fine-Grained, Lower Impact

| Parameter | Default | Informant | Effect |
|-----------|---------|-----------|--------|
| `compoundConfidence` | 0.6 | CompoundAgentInformant | Compound word weight |
| `specialNamedEntityConfidence` | 0.6 | CompoundAgentInformant | CamelCase/snake_case weight |
| `probability` (SeparatedNames) | 0.8 | SeparatedNamesInformant | Package-qualified name weight |
| `confidence` (CompoundRec) | 0.8 | CompoundRecommendationInformant | Compound recommendation weight |
| `nameOrTypeWeight` | 0.5 | Multiple informants | NAME/TYPE balance |

### Parameters Already at Maximum (1.0) — Can Only Be Lowered

| Parameter | Informant | Lowering effect |
|-----------|-----------|-----------------|
| `probability` | InDepArcsInformant | Weakens dependency-based extraction |
| `probability` | NameTypeInformant | Weakens pattern matching |
| `probability` | ExtractionDependentOccurrenceInformant | Weakens direct model matching |
| `probability` | InstantConnectionInformant | Weakens instant connections |
| `probability` | NameTypeConnectionInformant | Weakens pattern connections |

---

## How to Modify at Runtime

All `@Configurable` fields can be changed via ARDoCo's configuration system without code changes:

```java
// Example: adjust via AbstractConfigurable
Map<String, String> config = new TreeMap<>();
config.put("NounInformant:probability", "0.4");
config.put("MappingCombinerInformant:minCosineSimilarity", "0.6");
informant.applyConfiguration(config);
```

For `CommonTextToolsConfig.properties`, modify the file directly or override via system properties.

For hardcoded constants in `NerConnectionInformant`, a code change is required.

---

## Source File Locations

```
ardoco/tlr/stages-tlr/
├── text-extraction/src/main/java/.../textextraction/informants/
│   ├── CompoundAgentInformant.java
│   ├── NounInformant.java
│   ├── InDepArcsInformant.java
│   ├── OutDepArcsInformant.java
│   ├── SeparatedNamesInformant.java
│   └── MappingCombinerInformant.java
├── recommendation-generator/src/main/java/.../recommendationgenerator/informants/
│   ├── NameTypeInformant.java
│   ├── CompoundRecommendationInformant.java
│   └── ProjectNameInformant.java
├── connection-generator/src/main/java/.../connectiongenerator/informants/
│   ├── ExtractionDependentOccurrenceInformant.java
│   ├── InstantConnectionInformant.java
│   ├── NameTypeConnectionInformant.java
│   └── ReferenceInformant.java
└── connection-generator-ner/src/main/java/.../connectiongenerator/ner/informants/
    └── NerConnectionInformant.java

ardoco/core/framework/common/src/main/resources/configs/
    └── CommonTextToolsConfig.properties

ardoco/tlr/stages-tlr/recommendation-generator/src/main/java/.../recommendationgenerator/
    └── RecommendedInstanceImpl.java (aggregation logic)
```
