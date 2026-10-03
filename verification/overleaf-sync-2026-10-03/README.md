# Overleaf sync check, 2026-10-03

Before: ./scripts/sync-paper-overleaf.sh --check exit 1: 'Paper has uncommitted tracked changes' (remotes origin/overleaf both 0 0 vs HEAD; the guard worked as designed, no hook defect).
Action: committed paper edits (01affe4) with hooks enabled; post-commit pushed to origin/main and overleaf/main, then updated and pushed the parent pointer (320ec238).

## After
$ git -C paper rev-parse HEAD origin/main overleaf/main
01affe405888a3c949fbcecd55eabcf01710a5fc
01affe405888a3c949fbcecd55eabcf01710a5fc
01affe405888a3c949fbcecd55eabcf01710a5fc
rc=0
$ ./scripts/sync-paper-overleaf.sh --check
origin/main already contains 01affe405888a3c949fbcecd55eabcf01710a5fc.
overleaf/main already contains 01affe405888a3c949fbcecd55eabcf01710a5fc.
rc=0
