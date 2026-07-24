# Pre-orchestrator S24 archive

Archived on 2026-07-24 after `SLinker24RoleOrchestrator` became the sole
public and runnable S24 variant.

This archive preserves the superseded S21-floor augmentation designs:

- anchored recovery (`SLinker24`);
- one-shot agentic recovery (`SLinker24Agentic`);
- sequential dynamic recovery (`SLinker24Dynamic`);
- their dedicated pilots and contract tests;
- spikes 006–009 documenting the progression to the replacement orchestrator.

The earlier replacement-orchestrator implementation remains in the runtime
package only as the private `_SLinker24OrchestratorBase`, because the retained
role orchestrator subclasses and reuses its entity/coreference pipeline.

The active S24 research and verification trail starts at spike 010. The only
runner registry entry and package export is `s_linker24_role_orchestrator`.
