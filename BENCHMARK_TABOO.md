# Benchmark Taboo List

Terms from the 5 ARDoCo benchmark projects that MUST NOT appear in any LLM prompt.
Any example, analogy, or illustration in prompts must avoid these terms entirely.

## MediaStore
Components: UserDBAdapter, AudioWatermarking, Reencoding, MediaManagement, Facade, MediaAccess, Packaging, DB, FileStorage, TagWatermarking, Cache, UserManagement, DownloadLoadBalancer, ParallelWatermarking
Aliases: Database, DataStorage, ReEncoder, AudioAccess
Keywords: watermark, watermarking, reencoding, media, audio, facade, cache, packaging, adapter

## TeaStore
Components: WebUI, Registry, Persistence, Recommender, Auth, SlopeOneRecommender, OrderBasedRecommender, DummyRecommender, PopularityBasedRecommender, ImageProvider, PreprocessedSlopeOneRecommender
Aliases: PersistenceProvider, Image Provider, Web UI, UI (as TeaStore frontend)
Keywords: recommender, persistence, registry, auth, slope one, image provider, order (as in OrderBasedRecommender)

## Teammates
Components: Common, UI, Logic, Storage, Test Driver, E2E, Client, GAE Datastore
Aliases: Datastore, UI Component, Logic Component, Storage Component
Keywords: logic, storage, common, client, datastore, GAE

## BigBlueButton
Components: Recording Service, kurento, WebRTC-SFU, HTML5 Server, HTML5 Client, Presentation Conversion, BBB web, Redis PubSub, FSESL, Apps, Redis DB, FreeSWITCH
Aliases: KMS, Kurento Media Server, bbb-html5, bbb-web, Apps Akka, BigBlueButton Apps, fsels, Recording Processor, FreeSWITCH Event Socket Layer
Keywords: recording, kurento, freeswitch, redis, pubsub, conversion, bbb, html5, event (FSESL expansion), socket (FSESL expansion), layer (FSESL expansion), processor (Recording Processor alias)

## JabRef
Components: gui, cli, logic, globals, model, preferences
Aliases: GUI, command line interface, business logic
Keywords: gui, cli, preferences, globals, bibdatabase, bibentry

## Universal Taboo (appears in multiple projects)
- logic (Teammates component + JabRef component)
- UI (Teammates + TeaStore)
- client (Teammates component)
- storage (Teammates component)
- common (Teammates component)
- model (JabRef component)
- database / DB (MediaStore)
- cache (MediaStore)
- registry (TeaStore)
- auth (TeaStore)
- server (BBB)
- persistence (TeaStore)
- facade (MediaStore)
- recording (BBB)
- cascade (Teammates — "cascade logic", "cascade delete")
- conversion (BBB — Presentation Conversion component)
- validation (Teammates — "input validation")
- dedicated (MediaStore — "dedicated file server")
- preferences (JabRef component)
- config (BBB — bbb config files)
- internal (BBB/Teammates — "X.internal module")
- adapter (MediaStore — UserDBAdapter component word)
- order (TeaStore — OrderBasedRecommender component word)
- processor (BBB — Recording Processor alias word)
- event (BBB — FreeSWITCH Event Socket Layer alias word)
- socket (BBB — FreeSWITCH Event Socket Layer alias word)
- layer (BBB — FreeSWITCH Event Socket Layer alias word)

## Safe SE Textbook Examples (confirmed not in benchmark)
Use these domains for prompt examples:
- Compiler design: Lexer, Parser, AST, CodeGenerator, Optimizer, SymbolTable
- Operating systems: Scheduler, MemoryManager, FileSystem, ProcessTable, Dispatcher
- Networking: Router, Multiplexer, PacketHandler
- E-commerce (generic): ShoppingCart, PaymentGateway, InvoiceHandler, InventoryTracker
- Version control: Repository, CommitLog, BranchManager, MergeResolver
- Game engine: RenderEngine, PhysicsSimulator, InputHandler, SceneGraph
- Middleware: Broker, Wrapper, Connector (ambiguous examples)

## Tailored Code Anti-Patterns (NEVER do this)

Tailoring code paths to benchmark-specific casing, naming, or surface forms is the same
class of leakage as putting benchmark terms in prompts. Two illustrative anti-patterns
surfaced during EXT-01 — both are banned.

### Anti-pattern: Case-mismatch regex baselines
Building `re.compile(r"\b{name}\b", flags=0)` from a component string that happens to be
lowercase in one project (`kurento`) but appears Capitalized in its documentation
(`Kurento`) silently encodes a project-specific casing convention. The regex returns ∅
for the documented form. Any downstream "is `X` mentioned standalone?" check inherits the
blind spot.

- Symptom: per-(component, dataset) anchor-set is empty for a component that *is*
  mentioned in the docs, producing Jaccard = 0 artefacts in diff stages.
- Fix path: **do not** patch the regex with per-component `flags=re.IGNORECASE` or
  per-component casing tables — that bakes the benchmark casing convention into code.
  Replace the structural check with a project-agnostic LLM primitive (EXT-01 pattern)
  that handles casing variation as a natural-language detail, not a regex flag.
- Detection: any per-component regex pattern, per-component flag override, or
  per-component synonym map. Audit by grepping for `re.IGNORECASE` near component
  iteration and for hard-coded casing variants.

### Anti-pattern: Tailoring diff/comparison rules to specific (component, dataset) cells
Adjusting a Jaccard or symmetric-difference threshold to "rescue" one failing cell
(e.g., relaxing `min_jaccard_per_comp` only when the offending component is `kurento`)
shifts the leak from prompt to threshold logic. Same class of leakage, different layer.

- Fix path: if a rule fires on what is provably a baseline blind spot (not a real
  divergence in the variant under test), drop the baseline-as-ground-truth assumption
  for that cell and document the inspection. Do not retune the rule per benchmark term.

