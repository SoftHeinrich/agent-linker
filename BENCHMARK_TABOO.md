# Benchmark Taboo List

Terms from the 5 ARDoCo benchmark projects that MUST NOT appear in any LLM prompt.
Any example, analogy, or illustration in prompts must avoid these terms entirely.

## MediaStore
Components: UserDBAdapter, AudioWatermarking, Reencoding, MediaManagement, Facade, MediaAccess, Packaging, DB, FileStorage, TagWatermarking, Cache, UserManagement, DownloadLoadBalancer, ParallelWatermarking
Aliases: Database, DataStorage, ReEncoder, AudioAccess
Keywords: watermark, watermarking, reencoding, media, audio, facade, cache, packaging

## TeaStore
Components: WebUI, Registry, Persistence, Recommender, Auth, SlopeOneRecommender, OrderBasedRecommender, DummyRecommender, PopularityBasedRecommender, ImageProvider, PreprocessedSlopeOneRecommender
Aliases: PersistenceProvider, Image Provider, Web UI, UI (as TeaStore frontend)
Keywords: recommender, persistence, registry, auth, slope one, image provider

## Teammates
Components: Common, UI, Logic, Storage, Test Driver, E2E, Client, GAE Datastore
Aliases: Datastore, UI Component, Logic Component, Storage Component
Keywords: logic, storage, common, client, datastore, GAE

## BigBlueButton
Components: Recording Service, kurento, WebRTC-SFU, HTML5 Server, HTML5 Client, Presentation Conversion, BBB web, Redis PubSub, FSESL, Apps, Redis DB, FreeSWITCH
Aliases: KMS, Kurento Media Server, bbb-html5, bbb-web, Apps Akka, BigBlueButton Apps, fsels
Keywords: recording, kurento, freeswitch, redis, pubsub, conversion, bbb, html5

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

## Safe SE Textbook Examples (confirmed not in benchmark)
Use these domains for prompt examples:
- Compiler design: Lexer, Parser, AST, CodeGenerator, Optimizer, SymbolTable
- Operating systems: Scheduler, MemoryManager, FileSystem, ProcessTable, Dispatcher
- Networking: Router, LoadBalancer (careful - MediaStore has this), Socket, PacketHandler
- E-commerce (generic): ShoppingCart, PaymentGateway, OrderProcessor, InventoryTracker
- Version control: Repository, CommitLog, BranchManager, MergeResolver
- Game engine: RenderEngine, PhysicsSimulator, InputHandler, SceneGraph
