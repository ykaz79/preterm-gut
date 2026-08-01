import pandas as pd, numpy as np, warnings
import statsmodels.api as sm, statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

cst=pd.read_pickle("cst_v2.pkl"); meta=pd.read_pickle("meta.pkl")
g=pd.read_pickle("genus.pkl").set_index("siid"); f=pd.read_pickle("family.pkl").set_index("siid")
den=g[[c for c in g.columns if c not in ("Sample","total_all")]].sum(axis=1)
d=meta.merge(cst,on="siid")
d["C2"]=(d.cluster==1).astype(int)
d["K"]=d.siid.map(g["Klebsiella"]/den*100)
d["E"]=d.siid.map(f["Enterobacteriaceae"]/den*100)
d["post"]=((d.affected==1)&(d.dol>=d.ae_dol)); d=d[~d.post].copy()
d["y"]=((d.affected==1)&(d.dol<d.ae_dol)).astype(int)

print("【1】C2 は Klebsiella 相対量とどれだけ同じ情報か")
print(f"  Klebsiella% で C2 所属を判別: AUC = {roc_auc_score(d.C2, d.K):.4f}")
print(f"  Enterobacteriaceae% で C2 所属を判別: AUC = {roc_auc_score(d.C2, d.E):.4f}")
print(f"  Spearman r (Klebsiella% vs C2): {d[['K','C2']].corr('spearman').iloc[0,1]:.3f}")
thr=np.percentile(d.K,100*(1-d.C2.mean()))
print(f"  『Klebsiella ≥ {thr:.1f}%』という単純な閾値で C2 を再現した場合の一致率: "
      f"{((d.K>=thr).astype(int)==d.C2).mean()*100:.1f}%  (κ = "
      f"{(lambda a,b: (( (a==b).mean() - ((a.mean()*b.mean())+((1-a.mean())*(1-b.mean()))) ) / (1-((a.mean()*b.mean())+((1-a.mean())*(1-b.mean()))))))((d.K>=thr).astype(int), d.C2):.3f})")

print("\n【2】発症前の判別能: 連続量 vs C2 二値（検体レベル, n=197）")
for lab,x in [("Klebsiella% (連続)",d.K), ("Enterobacteriaceae% (連続)",d.E), ("C2 所属 (二値)",d.C2)]:
    print(f"  {lab:30s} AUC = {roc_auc_score(d.y, x):.4f}")

print("\n【3】互いに調整した場合（GEE, 症例クラスタ, 交換可能）")
for fml in ["y ~ K", "y ~ C2", "y ~ K + C2"]:
    m=smf.gee(fml, groups=d["case"], data=d, family=sm.families.Binomial(),
              cov_struct=sm.cov_struct.Exchangeable()).fit()
    terms=", ".join(f"{k}: OR {np.exp(v):.2f} (p={m.pvalues[k]:.3f})"
                    for k,v in m.params.items() if k!="Intercept")
    print(f"  {fml:14s} → {terms}")

print("\n【4】C2 が Klebsiella 以外に含む情報")
P=(d.set_index("siid"))
prof=pd.DataFrame({"C2内 平均%":[d.loc[d.C2==1,c].mean() if c in d else np.nan for c in ["K","E"]]},
                  index=["Klebsiella","Enterobacteriaceae(科)"])
print(prof.round(1).to_string())
print(f"  C2 検体における Klebsiella 以外の Enterobacteriaceae: "
      f"{(d.loc[d.C2==1,'E']-d.loc[d.C2==1,'K']).mean():.1f}%")
print(f"  C2 検体で Klebsiella が最優占でない検体: "
      f"{int((d.loc[d.C2==1,'K'] < d.loc[d.C2==1,'E']-d.loc[d.C2==1,'K']).sum())}/{int(d.C2.sum())}")

print("\n【5】縦断情報（C2 への移行・滞留）は相対量に還元できるか")
inf=d.groupby("case").agg(y=("y","max"), Kmax=("K","max"), Kmean=("K","mean"),
                          everC2=("C2","max"), nC2=("C2","sum"))
print(f"  児レベル AUC  Klebsiella最大値: {roc_auc_score(inf.y, inf.Kmax):.3f}   "
      f"Klebsiella平均: {roc_auc_score(inf.y, inf.Kmean):.3f}   "
      f"C2 回数: {roc_auc_score(inf.y, inf.nC2):.3f}   C2 経験有無: {roc_auc_score(inf.y, inf.everC2):.3f}")
