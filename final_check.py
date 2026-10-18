"""Check that every number in the paper matches the result files in reports/.

Run from the repository root: python final_check.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PAPER = Path("paper")
TEX = PAPER / "main.tex"
t = TEX.read_text(encoding="utf-8")
flat = re.sub(r"\s+", "", t)
ok_all: list[bool] = []


def chk(label: str, needle: str) -> None:
    hit = re.sub(r"\s+", "", needle) in flat
    ok_all.append(hit)
    if not hit:
        print(f"  [MISS] {label:44s} {needle}")


def load(p: str) -> dict:
    return json.loads(Path(p).read_text())


print("=" * 72)
print("1. NUMBERS vs ARTEFACTS")
print("=" * 72)

s = load("reports/final_results_table.json")["summary"]
for name, v in s.items():
    for met in ["hr@1", "hr@5", "hr@10", "ndcg@10", "mrr"]:
        chk(f"{name} {met}", f"{v[met]:.4f}")
        chk(f"{name} {met} sd", f"({v[met + '_sd']:.4f})")

g = load("reports/final_results_table.json")["aligned_minus_popularity"]
for k in ["mean", "sd", "min", "max"]:
    chk(f"gap {k}", f"{g[k]:.4f}")

orc = load("reports/oracle_retrieval_results.json")["results"]
for p in ["0.0", "1.0"]:
    for met in ["hr@10", "mrr"]:
        chk(f"oracle p={p} {met}", f"{orc[p][met]:.4f}")

abl = load("reports/budget_ablation_results.json")["results"]
for b in ["50", "100", "150", "200", "300", "500"]:
    r = abl[b]
    chk(f"budget {b} recall", f"{r['cand']['recall'] * 100:.2f}")
    chk(f"budget {b} aligned", f"{r['cand']['hr@10']:.4f}")
    chk(f"budget {b} obs", f"{r['obs']['hr@10']:.4f}")

ml = load("reports/ml_density_check.json")
chk("ML ratings", f"{ml['ratings']:,}".replace(",", "{,}"))
chk("ML items", f"{ml['items']:,}".replace(",", "{,}"))
chk("ML density", f"{ml['density_pct']:.4f}")
chk("ML recall", f"{ml['recall_at_200_pct']:.2f}")

for label, val in [("interactions", "898{,}387"), ("users", "79{,}838"),
                   ("items", "48{,}154"), ("density", "0.0234"),
                   ("train", "721{,}704"), ("test", "176{,}683")]:
    chk(label, val)

for label, val in [("pop obs", "31.3"), ("pop aligned", "84.3"),
                   ("svd aligned", "0.2"), ("residual", "15.7")]:
    chk(label, val)

print(f"  {sum(ok_all)}/{len(ok_all)} numeric checks passed")

print("\n" + "=" * 72)
print("2. STRUCTURE")
print("=" * 72)
for env in ["table", r"table\*", "figure", r"figure\*", "tabularx",
            "equation", "align", "abstract", "document"]:
    b = len(re.findall(r"\\begin\{" + env + r"\}", t))
    e = len(re.findall(r"\\end\{" + env + r"\}", t))
    flag = "OK" if b == e else "*** MISMATCH ***"
    print(f"  {env:12s} {b:2d}/{e:<2d} {flag}")
print(f"  braces delta: {t.count('{') - t.count('}')}")

labs = set(re.findall(r"\\label\{([^}]*)\}", t))
refs = set(re.findall(r"\\ref\{([^}]*)\}", t))
print(f"  dangling \\ref: {sorted(refs - labs) or 'none'}")

print("\n" + "=" * 72)
print("3. FIGURES")
print("=" * 72)
for f in re.findall(r"\\includegraphics\[[^]]*\]\{([^}]*)\}", t):
    p = PAPER / "figures" / f
    print(f"  {'OK  ' if p.exists() else 'MISS'} {f}")

print("\n" + "=" * 72)
print("4. BIBLIOGRAPHY")
print("=" * 72)
bib = (PAPER / "references.bib").read_text(encoding="utf-8")
used: list[str] = []
for grp in re.findall(r"\\cite[tp]\{([^}]*)\}", t):
    used += [x.strip() for x in grp.split(",") if x.strip()]
have = set(re.findall(r"@\w+\{([^,]+),", bib))
print(f"  cited {len(set(used))}, defined {len(have)}")
print(f"  missing from bib : {sorted(set(used) - have) or 'none'}")
print(f"  defined unused   : {sorted(have - set(used)) or 'none'}")
nodoi = [k for typ, k, body in re.findall(r"@(\w+)\{([^,]+),(.*?)\n\}", bib, re.S)
         if "doi" not in body.lower()]
print(f"  without DOI ({len(nodoi)}): {nodoi}")

print("\n" + "=" * 72)
print("5. ARXIV READINESS")
print("=" * 72)
print(f"  \\documentclass    : {re.search(r'documentclass(.*)', t).group(1)}")
pkgs = re.findall(r"\\usepackage(?:\[[^]]*\])?\{([^}]*)\}", t)
print(f"  packages ({len(pkgs)}): {', '.join(pkgs)}")
risky = [p for p in pkgs if p in {"minted", "pythontex", "shellesc"}]
print(f"  shell-escape pkgs : {risky or 'none (good)'}")
n_inc = len(re.findall(r"\\(?:input|include)\{", t))
print(f"  input/include     : {n_inc} (0 is simplest for arXiv)")
figs = re.findall(r"includegraphics\[[^]]*\]\{([^}]*)\}", t)
print(f"  figure formats    : {sorted(set(Path(f).suffix for f in figs))}")
print(f"  \\bibliography     : {'yes, ships .bbl or .bib' if 'bibliography{' in t else 'MISSING'}")
