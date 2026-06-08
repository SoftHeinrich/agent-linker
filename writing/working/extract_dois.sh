#!/usr/bin/env bash
# Extract all DOIs from a BibTeX file into a single block.
# Each line:  <doi>    # <citekey>
#
# Usage:
#   ./extract_dois.sh                       # defaults to smelly-discussion.bib
#   ./extract_dois.sh path/to/file.bib
#   ./extract_dois.sh path/to/file.bib > dois.txt
#
# Note: walks entries top-down, so the DOI on each line is paired with
# the cite key of the entry it lives in. Entries without a DOI are
# emitted as "NO-DOI  # <citekey>" so they are easy to spot.

set -euo pipefail

BIB="${1:-$(dirname "$0")/smelly-discussion.bib}"

if [[ ! -f "$BIB" ]]; then
  echo "ERROR: bib file not found: $BIB" >&2
  exit 1
fi

awk '
  # Match @type{key,   — capture the cite key
  /^@[a-zA-Z]+\{[^,]+,/ {
    if (key != "" && !found_doi) {
      printf "NO-DOI                                              # %s\n", key
    }
    # Extract key between { and ,
    line = $0
    sub(/^@[a-zA-Z]+\{/, "", line)
    sub(/,.*$/, "", line)
    key = line
    found_doi = 0
    next
  }
  # Match a doi field (case-insensitive) — capture the DOI, strip
  # braces/quotes/whitespace. Handles all of:
  #   doi = {10.1234/foo},   DOI = "10.1234/foo",
  #   doi={10.1234/foo}}  (no trailing comma, entry closes on the same line).
  tolower($0) ~ /^[[:space:]]*doi[[:space:]]*=/ {
    line = $0
    sub(/^[^=]*=[[:space:]]*/, "", line)         # strip up to and incl. =
    gsub(/[{}",[:space:]]/, "", line)             # strip braces, quotes, ws
    sub(/,$/, "", line)                            # strip optional trailing ,
    printf "%-52s# %s\n", line, key
    found_doi = 1
    next
  }
  END {
    if (key != "" && !found_doi) {
      printf "NO-DOI                                              # %s\n", key
    }
  }
' "$BIB"
