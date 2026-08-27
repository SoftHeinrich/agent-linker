# Provenance — `jabref-maindeps-file.json.gz`

A file-level dependency graph of JabRef's main source, produced once by the
open-source **Depends** tool. It is bundled so the analysis reproduces on a bare
`python3` without cloning JabRef or running Java. Regenerate only if the benchmark's
JabRef revision changes.

## Source revision (pinned)

- Repo/commit: `https://github.com/ArDoCo/jabref` @ `6269698cae437610ec79c38e6dd611eef7e88afe`
- This is the exact commit named in the benchmark at
  `benchmark/jabref/model_2023/code/README.md`, so file paths match
  `goldstandard_sam_2021-code_2023.csv`.
- Scope analysed: `src/main/java` (1450 `.java` files → 1451 graph nodes, 9089 edges).

## Tool

- **Depends** v0.9.7 (multilang-depends), open-source Java source analyzer (no build needed).
  <https://github.com/multilang-depends/depends>

## Exact regeneration steps

```bash
# 1. fetch the pinned JabRef revision (shallow)
mkdir jabref && cd jabref && git init -q
git remote add origin https://github.com/ArDoCo/jabref.git
git fetch --depth 1 origin 6269698cae437610ec79c38e6dd611eef7e88afe
git checkout -q FETCH_HEAD

# 2. get Depends
curl -sSL -o depends.zip \
  https://github.com/multilang-depends/depends/releases/download/v0.9.7/depends-0.9.7-package-20221104a.zip
unzip -q depends.zip            # -> depends-0.9.7/depends.jar

# 3. produce the file-level JSON graph (edge src->dest means "src depends on dest")
java -Xmx4g -jar depends-0.9.7/depends.jar --auto-include -d <outdir> \
     -g file -f json java src/main/java jabref-maindeps
#    -> <outdir>/jabref-maindeps-file.json     (then gzip into data/)
```

The graph JSON schema: `{"variables": [<file paths>], "cells": [{"src": i, "dest": j,
"values": {<DepType>: weight}}]}`. `depimport.py` reads it directly (gzipped).

## Edge semantics

`src` depends on `dest`. Afferent coupling (Ca) of a component counts distinct external
files that appear as `src` pointing into the component; efferent (Ce) counts distinct
external `dest` the component points to.
