import pandas as pd, numpy as np
from scipy import stats
cst=pd.read_pickle("cst_v2.pkl"); meta=pd.read_pickle("meta.pkl"); g=pd.read_pickle("genus.pkl")
NAME={7:"C1",1:"C2",4:"C3",2:"C4",0:"C5",5:"C6",6:"C7",3:"C8"}
d=meta[meta.siid.isin(set(g.siid))].merge(cst,on="siid")
d["PGCT"]=d.cluster.map(NAME)
order=[f"C{i}" for i in range(1,9)]

print("="*70); print("【1】検体レベル: 施設 × PGCT（全206検体）"); print("="*70)
ct=pd.crosstab(d.PGCT, d.site).reindex(order)
ct["計"]=ct.sum(1); ct.loc["計"]=ct.sum()
print(ct.to_string())
core=pd.crosstab(d.PGCT,d.site).reindex(order)
pct=(core.div(core.sum(0),axis=1)*100).round(1)
print("\n施設内での構成比(%)"); print(pct.to_string())
chi2,p,dof,exp=stats.chi2_contingency(core)
print(f"\nχ²={chi2:.1f}, df={dof}, P={p:.3g}  （期待度数<5: {(exp<5).sum()}/{exp.size}）")

print("\n"+"="*70); print("【2】児レベル: 各PGCTを1回以上経験した児の数"); print("="*70)
rows=[]
for t in order:
    ever=d.assign(f=(d.PGCT==t)).groupby("case").agg(f=("f","max"), site=("site","first"))
    a=int(((ever.site=="A")&ever.f).sum()); b=int(((ever.site=="B")&ever.f).sum())
    ft=stats.fisher_exact([[a,19-a],[b,24-b]])
    rows.append(dict(PGCT=t, SiteA=f"{a}/19 ({a/19*100:.0f}%)", SiteB=f"{b}/24 ({b/24*100:.0f}%)",
                     Fisher_P=round(ft.pvalue,4)))
print(pd.DataFrame(rows).to_string(index=False))

print("\n"+"="*70); print("【3】プロバイオティクス関連型の施設分布"); print("="*70)
for t,desc in [("C7","Lacticaseibacillus 優位 = Site B のレジメン"),
               ("C1","Bifidobacterium/Clostridium"),("C3","Bifidobacterium"),
               ("C4","Bifidobacterium/Enterococcus"),("C5","Bifidobacterium/Staphylococcus")]:
    sub=core.loc[t]
    print(f"  {t} ({desc}): Site A {sub['A']}検体, Site B {sub['B']}検体")
bif=core.loc[["C1","C3","C4","C5"]].sum()
print(f"\n  Bifidobacterium 系4型(C1/C3/C4/C5) 合計: Site A {bif['A']}, Site B {bif['B']}")
print(f"  施設内比率: Site A {bif['A']/core['A'].sum()*100:.1f}%, Site B {bif['B']/core['B'].sum()*100:.1f}%")
ft=stats.fisher_exact([[bif['A'],core['A'].sum()-bif['A']],[bif['B'],core['B'].sum()-bif['B']]])
print(f"  Fisher P = {ft.pvalue:.3g}, OR = {ft.statistic:.2f}")

print("\n"+"="*70); print("【4】C2 の施設分布"); print("="*70)
c2=core.loc["C2"]
print(f"  Site A {c2['A']}/{core['A'].sum()} 検体 ({c2['A']/core['A'].sum()*100:.1f}%), "
      f"Site B {c2['B']}/{core['B'].sum()} 検体 ({c2['B']/core['B'].sum()*100:.1f}%)")
ft=stats.fisher_exact([[c2['A'],core['A'].sum()-c2['A']],[c2['B'],core['B'].sum()-c2['B']]])
print(f"  Fisher P = {ft.pvalue:.4f}, OR = {ft.statistic:.2f}")
ever=d.assign(f=(d.PGCT=="C2")).groupby("case").agg(f=("f","max"), site=("site","first"))
print(f"  児レベル: Site A {int(((ever.site=='A')&ever.f).sum())}/19, Site B {int(((ever.site=='B')&ever.f).sum())}/24")
