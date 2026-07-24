# Production checkpoint v1 failure

The identifier tool itself added 2 TP / 0 FP on BigBlueButton, but its feedback
caused the controller to finalize before running the still-applicable
coreference capability. The final route was entity -> role -> identifier ->
finalize.

This exposed a workflow stopping bug. The available-tool registry already
removes capabilities without project-profile evidence, so every remaining
capability is evidence-backed. The controller may order that set but must not
finalize while it is non-empty. The next checkpoint applies that invariant;
there is no project branch, step count, or score threshold.
