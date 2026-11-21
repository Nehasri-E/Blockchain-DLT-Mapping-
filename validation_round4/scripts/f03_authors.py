"""Task 4: is a defensible author-level dependence analysis feasible?

The question is NOT whether author strings can be split. It is whether they can
be resolved into IDENTITIES reliable enough to cluster on. Those are different
problems and the distinction decides the answer.

  string matching      two papers listing "Wang, Y" are grouped together.
                       Cheap, and wrong in a predictable direction: common
                       surnames with single initials collide heavily, so
                       clusters are too large and standard errors are inflated
                       for the wrong reason.

  disambiguation       two papers are grouped only when the evidence says the
                       same person wrote both. Needs affiliation, ORCID,
                       email, co-author graph or a resolved external identifier.
                       None of these are in this dataset.

This script measures how bad the collision problem actually is, rather than
asserting it, and reports the numbers that decide feasibility.
"""
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

ROOT = Path("/home/claude/LDA_Project")
R4 = ROOT / "validation_round4"
OUT = R4 / "outputs"

# papers_with_topics.csv already carries Authors; merging the master frame
# would produce Authors_x / Authors_y rather than a single column.
df = pd.read_csv(ROOT / "outputs" / "papers_with_topics.csv")
assert "Authors" in df.columns
YEARS = list(range(2019, 2026))
win = df[df["Year"].isin(YEARS)].copy()

n_total = len(df)
n_auth = int(df["Authors"].notna().sum())
print(f"Papers: {n_total}; with a non-null Authors field: {n_auth} ({n_auth/n_total:.1%})")


def split_authors(s):
    if not isinstance(s, str) or not s.strip():
        return []
    parts = re.split(r";|\s\band\b\s", s)
    out = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip().strip(",.")
        if p:
            out.append(p)
    return out


def normalise(name):
    """Surname + first initial, lowercased, accents stripped, punctuation removed.

    This is deliberately the STANDARD naive normalisation, so that the collision
    rate measured below is the collision rate a string-matching approach would
    actually suffer.
    """
    n = name.lower()
    n = (n.replace("ä", "a").replace("ö", "o").replace("ü", "u").replace("ß", "ss")
          .replace("é", "e").replace("è", "e").replace("á", "a").replace("í", "i")
          .replace("ó", "o").replace("ú", "u").replace("ç", "c").replace("ñ", "n"))
    n = re.sub(r"[^a-z,\s\-]", "", n)
    if "," in n:
        sur, given = n.split(",", 1)
    else:
        toks = n.split()
        sur, given = (toks[-1], " ".join(toks[:-1])) if len(toks) > 1 else (n, "")
    sur = sur.strip().replace(" ", "-")
    gi = given.strip()[:1]
    return f"{sur}_{gi}" if sur else ""


df["author_list"] = df["Authors"].map(split_authors)
df["n_authors"] = df["author_list"].str.len()
df["norm_authors"] = df["author_list"].map(lambda L: [normalise(a) for a in L if normalise(a)])

parsed = int((df["n_authors"] > 0).sum())
print(f"Papers with at least one parsable author: {parsed} ({parsed/n_total:.1%})")
print(f"Mean authors per paper: {df['n_authors'].mean():.2f}; "
      f"max {int(df['n_authors'].max())}")

all_norm = [a for L in df["norm_authors"] for a in L]
cnt = Counter(all_norm)
n_unique = len(cnt)
print(f"Author mentions: {len(all_norm)}; distinct normalised keys: {n_unique}")

single_initial_keys = sum(1 for k in cnt if len(k.split("_")[-1]) <= 1)
top = cnt.most_common(15)

full_names = {}
for L_raw, L_norm in zip(df["author_list"], df["norm_authors"]):
    for raw, nk in zip(L_raw, L_norm):
        full_names.setdefault(nk, set()).add(re.sub(r"[^a-zA-Z,\s\-]", "", raw).lower().strip())

ambiguous = {k: v for k, v in full_names.items() if len(v) > 1}
collide_mentions = sum(cnt[k] for k in ambiguous)
print(f"\nCOLLISION DIAGNOSTIC")
print(f"  normalised keys mapping to >1 distinct raw spelling: "
      f"{len(ambiguous)} of {n_unique} ({len(ambiguous)/n_unique:.1%})")
print(f"  author mentions affected: {collide_mentions} of {len(all_norm)} "
      f"({collide_mentions/len(all_norm):.1%})")
print(f"\n  most frequent normalised keys (prime collision candidates):")
for k, v in top[:10]:
    print(f"    {k:<22} {v:>3} mentions, {len(full_names[k])} distinct raw spellings")

win["norm_authors"] = win["Paper_ID"].map(
    dict(zip(df["Paper_ID"], df["norm_authors"])))
pairs = 0
auth_to_papers = {}
for pid, L in zip(win["Paper_ID"], win["norm_authors"]):
    for a in set(L or []):
        auth_to_papers.setdefault(a, []).append(pid)
multi = {a: p for a, p in auth_to_papers.items() if len(p) > 1}
print(f"\nIn the 2019-2025 window ({len(win)} papers):")
print(f"  normalised authors appearing on >1 paper: {len(multi)}")
print(f"  papers touched by at least one repeat author: "
      f"{len({p for ps in multi.values() for p in ps})}")

rows = [{
    "metric": "papers in corpus", "value": n_total},
    {"metric": "papers with non-null Authors", "value": n_auth},
    {"metric": "pct papers with usable author string", "value": round(100 * parsed / n_total, 1)},
    {"metric": "total author mentions", "value": len(all_norm)},
    {"metric": "distinct normalised author keys", "value": n_unique},
    {"metric": "keys with >1 distinct raw spelling", "value": len(ambiguous)},
    {"metric": "pct of keys that are ambiguous", "value": round(100 * len(ambiguous) / n_unique, 1)},
    {"metric": "pct of mentions affected by ambiguity",
     "value": round(100 * collide_mentions / len(all_norm), 1)},
    {"metric": "keys with only a single given initial", "value": single_initial_keys},
    {"metric": "ORCID / affiliation / email fields available", "value": 0},
    {"metric": "repeat authors in 2019-2025 window", "value": len(multi)},
]
pd.DataFrame(rows).to_csv(OUT / "F6_author_parsing_diagnostics.csv", index=False)
pd.DataFrame([{"normalised_key": k, "mentions": cnt[k],
               "n_distinct_raw_spellings": len(full_names[k]),
               "raw_spellings": " | ".join(sorted(full_names[k])[:6])}
              for k, _ in cnt.most_common(60)]).to_csv(
    OUT / "F7_author_collision_examples.csv", index=False)

feasible = (parsed / n_total > 0.95) and (len(ambiguous) / n_unique < 0.05)
verdict = {
    "author_strings_parsable": bool(parsed / n_total > 0.95),
    "pct_papers_parsable": round(100 * parsed / n_total, 1),
    "disambiguation_fields_available": [],
    "pct_normalised_keys_ambiguous": round(100 * len(ambiguous) / n_unique, 1),
    "reliable_disambiguation_feasible": bool(feasible),
    "multiway_clustering_feasible": bool(feasible),
    "decision": ("Author strings parse cleanly, but the dataset contains no "
                 "affiliation, ORCID, email or co-author-resolved identifier, so only "
                 "surname-plus-initial string matching is possible. That is name "
                 "matching, not disambiguation. Multiway clustering by journal and "
                 "author is therefore NOT performed: it would rest on identity claims "
                 "the data cannot support, and the direction of its error is not even "
                 "predictable, since name collisions inflate cluster sizes while "
                 "name variants fragment true clusters. Documented as a limitation."),
}
import json
(OUT / "F8_author_feasibility_verdict.json").write_text(json.dumps(verdict, indent=2))
print(f"\n{'='*100}")
print(f"VERDICT: reliable disambiguation feasible = {feasible}")
print(verdict["decision"])
