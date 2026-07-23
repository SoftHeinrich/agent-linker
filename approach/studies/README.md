Direct matcher and router decoupling study.

These scripts are standalone and do not modify the existing linker code.

Files:
- `common.py`: shared benchmark loading and scoring helpers.
- `direct_matcher_study.py`: direct matcher opportunity, candidate quality, and
  residual-gap analysis.
- `router_decouple_compare.py`: rule-router vs LLM-router comparison on the
  same direct matcher.
