import pandas as pd, numpy as np, warnings
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
rng=np.random.default_rng(20260726)

cst=pd.read_pickle("cst_v2.pkl"); meta=pd.read_pickle("meta.pkl")
g=pd.read_pickle("genus.pkl").set_index("siid"); f=pd.read_pickle("family.pkl").set_index("siid")
den=g[[c for c in g.columns if c not in ("Sample","total_all")]].sum(axis=1)
d=meta.merge(cst,on="siid")
d["C2"]=(d.cluster==1).astype(int)
d["K"]=d.siid.map(g["Klebsiella"]/den*100).values
d["E"]=d.siid.map(f["Enterobacteriaceae"]/den*100).values
d=d[~((d.affected==1)&(d.dol>=d.ae_dol))].reset_index(drop=True)
d["y"]=((d.affected==1)&(d.dol<d.ae_dol)).astype(int)

cases=sorted(d.case.unique()); nc=len(cases)
idx={c:np.flatnonzero((d.case==c).values) for c in cases}
Y=d.y.values; K=d.K.values; E=d.E.values; C2=d.C2.values

A=pd.DataFrame({"case":cases})
A["y"]=[Y[idx[c]].max() for c in cases]
A["Kmax"]=[K[idx[c]].max() for c in cases]
A["Kmean"]=[K[idx[c]].mean() for c in cases]
A["nC2"]=[C2[idx[c]].sum() for c in cases]
A["ever"]=[C2[idx[c]].max() for c in cases]

def safe(y,x):
    y=np.asarray(y)
    return roc_auc_score(y,x) if 0<y.sum()<len(y) else np.nan

B=4000
picks=[rng.integers(0,nc,nc) for _ in range(B)]
samp_idx=[np.concatenate([idx[cases[i]] for i in p]) for p in picks]

def ci_sample(target, pred):
    v=[]
    for ii in samp_idx:
        a=safe(target[ii], pred[ii])
        if np.isfinite(a): v.append(a)
    return np.percentile(v,[2.5,97.5])
def ci_infant(tcol, pcol):
    tv=A[tcol].values; pv=A[pcol].values; v=[]
    for p in picks:
        a=safe(tv[p], pv[p])
        if np.isfinite(a): v.append(a)
    return np.percentile(v,[2.5,97.5])

rows=[]
def add(sec,lab,point,ci):
    lo,hi=ci; rows.append(dict(Section=sec,Predictor=lab,AUC=round(point,3),CI=f"{lo:.2f}–{hi:.2f}"))

add("A","Klebsiella relative abundance (continuous)", safe(Y,K), ci_sample(Y,K))
add("A","Enterobacteriaceae relative abundance (continuous)", safe(Y,E), ci_sample(Y,E))
add("A","Assignment to C2 (binary)", safe(Y,C2), ci_sample(Y,C2))
add("B","Maximum Klebsiella abundance", safe(A.y,A.Kmax), ci_infant("y","Kmax"))
add("B","Mean Klebsiella abundance", safe(A.y,A.Kmean), ci_infant("y","Kmean"))
add("B","Number of samples assigned to C2", safe(A.y,A.nC2), ci_infant("y","nC2"))
add("B","Ever assigned to C2 (binary)", safe(A.y,A.ever), ci_infant("y","ever"))
add("C","Klebsiella relative abundance", safe(C2,K), ci_sample(C2,K))
add("C","Enterobacteriaceae relative abundance", safe(C2,E), ci_sample(C2,E))

t=pd.DataFrame(rows); pd.set_option("display.width",210); pd.set_option("display.max_colwidth",56)
print(t.to_string(index=False))
print(f"\n検体レベル n={len(d)} (pre-onset {int(Y.sum())}, unaffected {int((Y==0).sum())}) / 児レベル n={nc} (affected {int(A.y.sum())})")
kap=lambda a,b: ((a==b).mean()-((a.mean()*b.mean())+((1-a.mean())*(1-b.mean()))))/(1-((a.mean()*b.mean())+((1-a.mean())*(1-b.mean()))))
thr=np.percentile(K,100*(1-C2.mean()))
print(f"閾値 Klebsiella >= {thr:.1f}% で C2 再現: 一致率 {((K>=thr).astype(int)==C2).mean()*100:.1f}%, kappa={kap((K>=thr).astype(int),C2):.3f}")
t.to_pickle("auc_tab.pkl")
