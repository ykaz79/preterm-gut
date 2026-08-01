import pandas as pd, numpy as np, pickle
from scipy import stats
D,ks,avg,msil,store=pickle.load(open("sil_cache.pkl","rb"))
X=pd.read_pickle("feature_table.pkl"); g=pd.read_pickle("genus.pkl").set_index("siid")
den=g[[c for c in g.columns if c not in ("Sample","total_all")]].sum(axis=1)
rel=X.div(den,axis=0)*100; rel=rel.where(rel>=0.01,0.0)
keep=[c for c in rel.columns if rel[c].mean()>=0.01 and (rel[c]>0).mean()>=0.10]
P=rel[keep]; P=P.div(P.sum(axis=1),axis=0)
meta=pd.read_pickle("meta.pkl"); m=meta[meta.siid.isin(set(P.index))].drop_duplicates("siid").set_index("siid")
m=m.reindex(P.index)
pre=((m.affected==1)&(m.dol<m.ae_dol)).values
post=((m.affected==1)&(m.dol>=m.ae_dol)).values
case=m.case.values; site=m.site.values
c2_k8=set(np.flatnonzero(store[8][1]==1))   # k=8 の C2

def retention(labels, target):
    dd=pd.DataFrame({"case":case,"t":m.t.values,"cl":labels})
    tot=hit=0
    for c,s in dd.groupby("case"):
        s=s.sort_values("t"); ts=s.t.tolist(); cl=s.cl.tolist()
        for i in range(len(s)-1):
            if ts[i+1]-ts[i]!=1: continue
            if cl[i] in target:
                tot+=1
                if cl[i+1] in target: hit+=1
    return hit,tot

rows=[]
for k in [6,7,8,9]:
    med,lab,sv=store[k]
    prof=(P*100).assign(cl=lab).groupby("cl").mean()
    kl=[c for c in range(k) if prof.loc[c].sort_values(ascending=False).index[0]
        in ("Klebsiella","Unclassified Enterobacteriaceae")]
    inKl=np.isin(lab,kl)
    ov=len(set(np.flatnonzero(inKl)) & c2_k8)
    # 検体レベル（発症後を除外）
    keepm=~post
    a=int((inKl&pre&keepm).sum()); b=int((inKl&~pre&keepm).sum())
    c_=int((~inKl&pre&keepm).sum()); dd_=int((~inKl&~pre&keepm).sum())
    fs=stats.fisher_exact([[a,b],[c_,dd_]])
    # 児レベル
    df=pd.DataFrame({"case":case,"in":inKl,"pre":pre,"post":post})
    df=df[~df.post]
    ev=df.groupby("case").agg(y=("pre","max"), ever=("in","max"))
    ct=pd.crosstab(ev.ever,ev.y)
    A=int(ct.loc[True,True]) if (True in ct.index and True in ct.columns) else 0
    B=int(ct.loc[True,False]) if (True in ct.index and False in ct.columns) else 0
    fi=stats.fisher_exact(ct.values) if ct.shape==(2,2) else None
    hit,tt=retention(lab,set(kl))
    kln=", ".join(f"n={int((lab==c).sum())} ({prof.loc[c].sort_values(ascending=False).index[0]} "
                  f"{prof.loc[c].sort_values(ascending=False).iloc[0]:.0f}%)" for c in kl)
    rows.append(dict(k=k, silhouette=round(avg[ks.index(k)],4), 型数=len(kl), 内訳=kln,
        検体数=int(inKl.sum()), k8C2との重複=f"{ov}/40",
        発症前=f"{a}/{a+c_}", 未発症=f"{b}/{b+dd_}", Fisher_P=f"{fs.pvalue:.4g}",
        児PPV=f"{A}/{A+B}" + (f" ({A/(A+B)*100:.0f}%)" if A+B else ""),
        児P=f"{fi.pvalue:.3f}" if fi else "\u2014",
        自己保持=f"{hit}/{tt} ({hit/tt*100:.1f}%)" if tt else "\u2014"))
t=pd.DataFrame(rows)
pd.set_option("display.width",250); pd.set_option("display.max_colwidth",52)
print(t.to_string(index=False))
