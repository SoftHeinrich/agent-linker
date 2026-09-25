# Entity-example illustration verification

Artifact: `paper/figures/drawio/coref-linker-illustration.drawio`

## Checks

```text
$ xmllint --noout paper/figures/drawio/coref-linker-illustration.drawio
PASS (exit 0; no output)

$ git diff --check -- paper/figures/drawio/coref-linker-illustration.drawio
PASS (exit 0; no output)

$ rg -n "entity_(accept|reject)|written=(alias|exact)|Approved alias" \
    paper/figures/drawio/coref-linker-illustration.drawio
PASS: accepted and rejected input/output cells are present; no
`written=alias` or `written=exact` value remains; the displayed mapping is
`Approved alias: DB -> database`.
```

The installed environment has no `drawio`/`draw.io` executable, so no rendered
export was produced. XML well-formedness and the S126 field/value assertions were
checked directly on the source artifact.
