"""Agreement between the check sheet's verdicts (human, or the declared model reader)
and the tiers. usage: check_agreement.py out/model_check_sheet.csv"""
import collections, csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
imply = {"gold": "A", "gold_plus_only": "A", "silver": "A", "refers": "R", "none": "N"}
tab = collections.defaultdict(collections.Counter)
for r in rows:
    v = (r.get("verdict(A/R/N)") or "").strip().upper()[:1]
    if v:
        tab[(r["tier"], r["pattern"])][v] += 1
print(f"{'tier':16s} {'pattern':13s} {'n':>3s} {'A':>3s} {'R':>3s} {'N':>3s}  agree-with-tier")
tot = ok = 0
per_tier = collections.defaultdict(lambda: [0, 0])
for (tier, pat), c in sorted(tab.items()):
    n = sum(c.values()); a = c[imply[tier]]
    tot += n; ok += a; per_tier[tier][0] += n; per_tier[tier][1] += a
    print(f"{tier:16s} {pat:13s} {n:3d} {c['A']:3d} {c['R']:3d} {c['N']:3d}  {a/n:.2f}")
print("per tier:", {t: f"{a}/{n}={a/n:.2f}" for t, (n, a) in per_tier.items()})
print(f"overall agreement with tier {ok}/{tot} = {ok/tot:.2f}")
# precision of the tiers as gold: gold/gold_plus pairs judged A; lenient counts R as acceptable-not-FP
for tier in ("gold", "gold_plus_only", "silver"):
    c = collections.Counter()
    for (t, _), cc in tab.items():
        if t == tier: c.update(cc)
    n = sum(c.values())
    if n: print(f"{tier}: judged A {c['A']/n:.2f}  A-or-R {(c['A']+c['R'])/n:.2f}  (n={n})")
