import pandas as pd, numpy as np
from scipy import stats

cst = pd.read_pickle("cst.pkl")
d   = pd.read_pickle("d_final.pkl")          # 197検体（発症後9検体を除外）
meta= pd.read_pickle("meta.pkl")
g   = pd.read_pickle("genus.pkl")

# cluster 1 = Klebsiella優占 = C2
C2 = 1
cst["is_C2"] = (cst.cluster == C2).astype(int)

full = meta[meta.siid.isin(set(g.siid))].merge(cst, on="siid")
full["pre"] = ((full.affected==1)&(full.dol<full.ae_dol))
full["post"] = ((full.affected==1)&(full.dol>=full.ae_dol))
ana = full[~full.post].copy()                # 197検体

print("=== C2 検体の帰属（発症後検体を除外した197検体）===")
t = pd.crosstab(ana.is_C2, np.where(ana.affected==1,"発症児(発症前)","未発症児"))
print(t.to_string())
print(f"\nC2検体 {int(ana.is_C2.sum())}件のうち、未発症児由来 "
      f"{int(ana[(ana.is_C2==1)&(ana.affected==0)].shape[0])}件 "
      f"({ana[(ana.is_C2==1)&(ana.affected==0)].shape[0]/ana.is_C2.sum()*100:.1f}%)")
print(f"（全206検体では C2 {int(full.is_C2.sum())}件, うち未発症児由来 "
      f"{int(full[(full.is_C2==1)&(full.affected==0)].shape[0])}件）")

def wilson(k, n, z=1.96):
    if n==0: return (np.nan, np.nan)
    p=k/n; den=1+z*z/n
    c=(p+z*z/(2*n))/den; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return (max(0,c-h)*100, min(1,c+h)*100)

def block(name, df):
    inf = df.groupby("case").agg(outcome=("affected","max"),
                                 everC2=("is_C2","max"),
                                 site=("site","first")).reset_index()
    a = int(((inf.everC2==1)&(inf.outcome==1)).sum())   # C2+ 発症
    b = int(((inf.everC2==1)&(inf.outcome==0)).sum())   # C2+ 未発症
    c = int(((inf.everC2==0)&(inf.outcome==1)).sum())   # C2- 発症
    dd= int(((inf.everC2==0)&(inf.outcome==0)).sum())   # C2- 未発症
    print(f"\n{'='*62}\n{name}   （児 n = {len(inf)}）\n{'='*62}")
    tab = pd.DataFrame([[a,b,a+b],[c,dd,c+dd],[a+c,b+dd,len(inf)]],
        index=["C2 に1回以上割付","C2 に割付なし","計"],
        columns=["合併症あり","合併症なし","計"])
    print(tab.to_string())
    orr = stats.fisher_exact([[a,b],[c,dd]])
    print(f"\nFisher 正確検定: P = {orr.pvalue:.4f}   オッズ比 = "
          f"{'∞' if orr.statistic==np.inf else f'{orr.statistic:.2f}'}")
    rows=[]
    for lab, k, n in [("感度 (sensitivity)", a, a+c),
                      ("特異度 (specificity)", dd, b+dd),
                      ("陽性的中率 (PPV)", a, a+b),
                      ("陰性的中率 (NPV)", dd, c+dd)]:
        lo,hi = wilson(k,n)
        rows.append(dict(指標=lab, 値=f"{k}/{n} = {k/n*100:.1f}%" if n else "—",
                         CI=f"{lo:.1f}–{hi:.1f}%" if n else "—"))
    print(pd.DataFrame(rows).to_string(index=False))
    return inf

inf_all = block("A) 全43児（発症児は発症前検体のみでC2判定）", ana)
inf_B   = block("B) Site B 24児", ana[ana.site=="B"])
_       = block("C) 参考: 全206検体（発症後検体も含めてC2判定）", full)

print("\n=== C2 に割り付けられた未発症児（Site B）===")
sub = ana[(ana.site=="B")]
inf = sub.groupby("case").agg(outcome=("affected","max"), everC2=("is_C2","max"),
                              nC2=("is_C2","sum"), n=("siid","size")).reset_index()
print(inf[(inf.everC2==1)].sort_values("outcome", ascending=False).to_string(index=False))
