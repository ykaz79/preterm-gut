import pandas as pd, numpy as np
pd.set_option("display.width", 200)

meta = pd.read_pickle("meta.pkl"); g = pd.read_pickle("genus.pkl"); f = pd.read_pickle("family.pkl")

# 欠番の確認
print("=== 検体番号の欠番 ===")
print("1-252のうち研究ノートに無い番号:", sorted(set(range(1,253)) - set(meta.siid)))
print("提出252 / ノート記載", len(meta), "/ 解析済", len(g), "→ シーケンス失敗", 252-len(g))

# バッチが症例内で一定か
m = meta.copy()
nb = m.groupby("case")["batch"].nunique()
print("\n=== 症例あたりのバッチ数 ===")
print(nb.value_counts().sort_index().rename("症例数"))
print("複数バッチにまたがる症例:", (nb>1).sum(), "/ 43")

# ---------- 相対量 ----------
GEN5 = ["Klebsiella","Escherichia","Enterobacter","Serratia"]
gcols = [c for c in g.columns if c not in ("Sample","siid","total_all")]
# 原稿の前処理: "Unclassified hit" は除外
den_cols = [c for c in gcols if c != "Unclassified hit"]
g = g.set_index("siid"); f = f.set_index("siid")
tot = g[den_cols].sum(axis=1)

rel = pd.DataFrame(index=g.index)
for c in GEN5:
    rel[c] = g[c] / tot * 100
rel["Family_Entero"] = f["Enterobacteriaceae"] / tot * 100
rel["unclassified_Entero"] = rel["Family_Entero"] - rel[GEN5].sum(axis=1)
rel["Summed_Entero"] = rel["Family_Entero"]

d = meta.merge(rel.reset_index(), on="siid", how="inner")
d["pre_onset"] = ((d.affected==1) & (d.dol < d.ae_dol)).astype(int)
d["at_after"]  = ((d.affected==1) & (d.dol >= d.ae_dol)).astype(int)
d["grp"] = np.where(d.pre_onset==1, "pre-onset", np.where(d.affected==1, "at/after", "unaffected"))
d = d[d.grp != "at/after"].copy()          # 原稿と同じく発症後検体を除外
d["case_pre"] = (d.grp=="pre-onset").astype(int)

print("\n=== 群構成 ===")
print(d.grp.value_counts())
print("pre-onset 検体の症例別内訳:")
print(d[d.grp=="pre-onset"].groupby("case").agg(n=("siid","size"), onset=("ae_dol","first"),
                                                dols=("dol", lambda s: sorted(s.tolist()))))

# ---------- 原稿 Table 2 の再現 ----------
def cliffs(a, b):
    a = np.asarray(a); b = np.asarray(b)
    gt = (a[:,None] > b[None,:]).sum(); lt = (a[:,None] < b[None,:]).sum()
    return (gt - lt) / (len(a)*len(b))
def cohend(a, b):
    a = np.asarray(a); b = np.asarray(b)
    s = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return (a.mean()-b.mean())/s

print("\n=== 原稿 Table 2 の再現 ===")
rowsout = []
for tax in ["Summed_Entero","Klebsiella","Escherichia","unclassified_Entero","Enterobacter","Serratia"]:
    a = d.loc[d.grp=="pre-onset", tax]; b = d.loc[d.grp=="unaffected", tax]
    rowsout.append(dict(taxon=tax,
        pre_med=f"{a.median():.2f} ({a.quantile(.25):.2f}-{a.quantile(.75):.2f})",
        un_med=f"{b.median():.2f} ({b.quantile(.25):.2f}-{b.quantile(.75):.2f})",
        cliff=round(cliffs(a,b),2), cohen=round(cohend(a,b),1)))
print(pd.DataFrame(rowsout).to_string(index=False))
print("""
原稿記載値:
 Summed Entero   pre 44.64 (33.51-71.28)  un 0.20 (0.00-18.38)  δ=0.63  d=1.4
 Klebsiella      pre 32.58 (0.28-55.74)   un 0.06 (0.00-0.30)   δ=0.62  d=1.8
 Escherichia     pre  0.02 (0.00-0.07)    un 0.00 (0.00-0.05)   δ=0.10  d=0.4
 unclass Entero  pre  0.34 (0.17-0.76)    un 0.03 (0.00-0.21)   δ=0.49  d=0.2
 Enterobacter    pre  0.00 (0.00-0.01)    un 0.00 (0.00-0.02)   δ=0.04  d=-0.2
 Serratia        pre  0.00 (0.00-0.00)    un 0.00 (0.00-0.00)   δ=-0.07 d=-0.1""")

d.to_pickle("d.pkl")
