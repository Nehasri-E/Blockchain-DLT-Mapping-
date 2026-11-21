import pandas as pd
import numpy as np
import io
import re
import ftfy
from pathlib import Path

RAW = Path("/root/.claude/uploads/b86f9bac-1ee2-5191-9fe6-401f33bd8b7b/d42072f9-EDA_INPUT.csv")
OUT = Path("/home/claude/blockchain_lda/data")
OUT.mkdir(parents=True, exist_ok=True)

MIN_ABSTRACT_WORDS = 50
MIN_CONCLUSION_WORDS = 50

log = []


def note(msg):
    log.append(msg)
    print(msg)


def hybrid_decode(path):
    b = path.read_bytes()
    parts = []
    i = 0
    while i < len(b):
        try:
            parts.append(b[i:].decode("utf-8"))
            break
        except UnicodeDecodeError as e:
            parts.append(b[i:i + e.start].decode("utf-8"))
            parts.append(b[i + e.start:i + e.end].decode("cp1252", errors="replace"))
            i = i + e.end
    return "".join(parts)


def repair(text):
    if not isinstance(text, str):
        return ""
    out = text
    for _ in range(3):
        fixed = ftfy.fix_text(out)
        if fixed == out:
            break
        out = fixed
    out = out.replace(" ", " ").replace("�", "")
    out = re.sub(r"\s+", " ", out).strip()
    return out


def words(s):
    return s.fillna("").astype(str).str.split().str.len()


raw_text = hybrid_decode(RAW)
df = pd.read_csv(io.StringIO(raw_text))
note(f"Loaded raw file: {df.shape[0]} rows x {df.shape[1]} columns")

df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
note(f"Dropped empty Unnamed columns -> {df.shape[1]} columns")

df["Year_Parsed"] = pd.to_numeric(df["Year"], errors="coerce")
spill = df["Year_Parsed"].isna()
note(f"Excel cell-overflow spill rows removed: {int(spill.sum())}")
df = df.loc[~spill].reset_index(drop=True)
df["Year"] = df["Year_Parsed"].astype(int)
df = df.drop(columns=["Year_Parsed"])

df["Title_Clean"] = df["Article Title (unified)"].map(repair)
df["Abstract_Clean"] = df["Abstract"].map(repair)
df["Conclusion_Clean"] = df["Conclusion"].map(repair)

suspect = re.compile(r"Ã|â€|Â|�")
before = sum(
    df[c].fillna("").astype(str).str.contains(suspect).sum()
    for c in ["Article Title (unified)", "Abstract", "Conclusion"]
)
after = sum(
    df[c].str.contains(suspect).sum()
    for c in ["Title_Clean", "Abstract_Clean", "Conclusion_Clean"]
)
note(f"Mojibake-suspect cells: {before} before repair, {after} after")

df["Mojibake_Residual"] = (
    df["Title_Clean"].str.contains(suspect)
    | df["Abstract_Clean"].str.contains(suspect)
    | df["Conclusion_Clean"].str.contains(suspect)
)

df["Conclusion_Truncated"] = df["Conclusion"].fillna("").astype(str).str.len() >= 32000
note(f"Conclusions hitting the Excel 32,767-char cell limit: {int(df['Conclusion_Truncated'].sum())}")

canon = (
    df.assign(k=df["Source title"].fillna("").str.lower().str.strip())
    .groupby("k")["Source title"]
    .agg(lambda s: s.value_counts().idxmax())
)
df["Journal"] = df["Source title"].fillna("").str.lower().str.strip().map(canon)
note(f"Journal spellings collapsed: {df['Source title'].nunique()} -> {df['Journal'].nunique()}")

df["DOI"] = df["DOI_URL"].fillna("").astype(str).str.strip().str.lower()
df["Abstract_Words"] = words(df["Abstract_Clean"])
df["Conclusion_Words"] = words(df["Conclusion_Clean"])

df["Is_Correction"] = df["Title_Clean"].str.lower().str.startswith(
    ("correction to", "corrigendum", "erratum", "retraction")
)
note(f"Correction / corrigendum notices flagged: {int(df['Is_Correction'].sum())}")

df["Title_Key"] = df["Title_Clean"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()

df["_rank"] = df["Is_Correction"].astype(int) * 1000 - df["Conclusion_Words"].fillna(0)
df = df.sort_values("_rank").reset_index(drop=True)

dup_doi = df.duplicated(subset=["DOI"]) & df["DOI"].ne("")
dup_title = df.duplicated(subset=["Title_Key"]) & df["Title_Key"].ne("")
df["Duplicate_Record"] = dup_doi | dup_title
note(f"Duplicate records flagged: {int(dup_doi.sum())} by DOI, {int(dup_title.sum())} by title, {int(df['Duplicate_Record'].sum())} total")

df["Abstract_Valid"] = df["Abstract_Words"] >= MIN_ABSTRACT_WORDS
df["Conclusion_Valid"] = df["Conclusion_Words"] >= MIN_CONCLUSION_WORDS
df["Analysis_Ready"] = df["Abstract_Valid"] & ~df["Duplicate_Record"] & ~df["Is_Correction"]

note(f"Abstract shorter than {MIN_ABSTRACT_WORDS} words: {int((~df['Abstract_Valid']).sum())}")
note(f"Conclusion missing or shorter than {MIN_CONCLUSION_WORDS} words: {int((~df['Conclusion_Valid']).sum())}")

df["Paper_ID"] = df["Row_UID"].fillna("")
missing_id = df["Paper_ID"].eq("")
df.loc[missing_id, "Paper_ID"] = [f"GEN:{i + 1}" for i in range(int(missing_id.sum()))]
note(f"Paper_ID generated for records with no Row_UID: {int(missing_id.sum())}")
assert df["Paper_ID"].is_unique, "Paper_ID is not unique"

df = df.sort_values(["Year", "Paper_ID"]).reset_index(drop=True)

corpus = df.loc[df["Analysis_Ready"]].copy()


def build_text(row, use_conclusion):
    parts = [row["Title_Clean"], row["Abstract_Clean"]]
    if use_conclusion and row["Conclusion_Valid"]:
        parts.append(row["Conclusion_Clean"])
    return " ".join(p for p in parts if p)


corpus["Document_Text_A"] = corpus.apply(build_text, axis=1, use_conclusion=False)
corpus["Document_Text_B"] = corpus.apply(build_text, axis=1, use_conclusion=True)
corpus["Doc_Words_A"] = words(corpus["Document_Text_A"])
corpus["Doc_Words_B"] = words(corpus["Document_Text_B"])

note(f"Final analysis corpus: {len(corpus)} papers")
note(f"  with a valid conclusion (Corpus B uses all three fields): {int(corpus['Conclusion_Valid'].sum())}")
note(f"  abstract only inside Corpus B: {int((~corpus['Conclusion_Valid']).sum())}")

master_cols = [
    "Paper_ID", "Title_Clean", "Year", "DOI", "Journal", "Authors",
    "CABS_Rank", "ABDC_Rank", "Abstract_Clean", "Conclusion_Clean",
    "Document_Text_A", "Document_Text_B",
    "Abstract_Words", "Conclusion_Words", "Doc_Words_A", "Doc_Words_B",
    "Conclusion_Valid",
]
qc_cols = [
    "Paper_ID", "Title_Clean", "Year", "DOI", "Abstract_Words", "Conclusion_Words",
    "Abstract_Valid", "Conclusion_Valid", "Duplicate_Record", "Is_Correction",
    "Mojibake_Residual", "Conclusion_Truncated", "Analysis_Ready",
]

corpus[master_cols].to_parquet(OUT / "lda_master.parquet", index=False)
corpus[master_cols].to_csv(OUT / "lda_master.csv", index=False, encoding="utf-8")
df[qc_cols].to_csv(OUT / "quality_control.csv", index=False, encoding="utf-8")
df.to_parquet(OUT / "all_records_audited.parquet", index=False)

reconciliation = pd.DataFrame({
    "Stage": [
        "Raw rows in EDA_INPUT.csv",
        "Less Excel spill rows",
        "Less correction notices",
        "Less duplicate records",
        "Less abstracts under threshold",
        "Final analysis corpus",
    ],
    "Rows": [
        1003,
        len(df),
        len(df) - int(df["Is_Correction"].sum()),
        len(df) - int((df["Is_Correction"] | df["Duplicate_Record"]).sum()),
        len(corpus),
        len(corpus),
    ],
})
reconciliation.to_csv(OUT / "reconciliation.csv", index=False)
print()
print(reconciliation.to_string(index=False))

(OUT / "stage1_log.txt").write_text("\n".join(log), encoding="utf-8")
