# Failed exact-evidence formatting smoke

This fresh OpenAI GPT-5.6-terra, no-reasoning run completed profiling and entity
processing. The next controller decision cited exact document evidence wrapped
in literal quotation-mark characters, so the structural exact-substring check
failed.

The follow-up strips only surrounding quotation punctuation before exact
matching. It does not use fuzzy matching or semantic fallback.
