# Failed feedback-parity replay

This attempt exposed full recorded entity/coreference candidates and validator
decisions to the controller. It failed during the third controller decision:

```text
RuntimeError: LLM request failed in phase_24_orchestrator_3:
[Errno 7] Argument list too long: 'codex'
```

Cause: detailed tool outputs, including repeated source sentences, were nested
inside controller history and resent on every decision. The representation was
not viable for larger documents.

The follow-up keeps complete tool outputs as evidence but passes the controller
a normalized list of accepted and rejected `(sentence, component)` references.
This is a state representation change, not a numeric truncation or length gate.
