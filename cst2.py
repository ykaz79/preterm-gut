import pandas as pd, numpy as np, itertools
X = pd.read_pickle("feature_table.pkl")          # ND を科レベルに分解済み（Unclassified hit は既に除外）
g = pd.read_pickle("genus.pkl").set_index("siid")
tot_all = g[[c for c in g.columns if c not in ("Sample","total_all")]].sum(axis=1)  # 分母は全リード

rel = X.div(tot_all, axis=0)*100
rel[rel < 0.01] = 0.0                             # 検体内 0.01% 未満は absent
keep = [c for c in rel.columns if (rel[c].mean() >= 0.01) and ((rel[c] > 0).mean() >= 0.10)]
print("フィルタ後 特徴量数:", len(keep), "（原稿記載: 21）")
for c in sorted(keep, key=lambda c: -rel[c].mean()):
    print(f"   {c:34s} mean {rel[c].mean():7.3f}%  prevalence {(rel[c]>0).mean()*100:5.1f}%")
P = rel[keep]; P = P.div(P.sum(axis=1), axis=0)

def jsd(P, sqrt=True):
    A=P.values; n=len(A); D=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            p,q=A[i],A[j]; m=0.5*(p+q)
            a=p>0; b=q>0
            d=0.5*np.sum(p[a]*np.log2(p[a]/m[a]))+0.5*np.sum(q[b]*np.log2(q[b]/m[b]))
            d=max(d,0.0); D[i,j]=D[j,i]=np.sqrt(d) if sqrt else d
    return D

def pam(D,k):
    n=len(D); med=[int(np.argmin(D.sum(1)))]
    while len(med)<k:
        cur=D[:,med].min(1); best,bg=None,-np.inf
        for c in range(n):
            if c in med: continue
            gain=np.maximum(cur-D[:,c],0).sum()
            if gain>bg: bg,best=gain,c
        med.append(best)
    imp=True
    while imp:
        imp=False; cost=D[:,med].min(1).sum()
        for mi in range(k):
            for h in range(n):
                if h in med: continue
                tr=med.copy(); tr[mi]=h; c=D[:,tr].min(1).sum()
                if c<cost-1e-12: med,cost,imp=tr,c,True
    return np.array(med), np.argmin(D[:,med],1)

def sil(D,lab):
    n=len(D); s=np.zeros(n)
    for i in range(n):
        same=(lab==lab[i]).copy(); same[i]=False
        if same.sum()==0: continue
        a=D[i,same].mean(); b=min(D[i,lab==c].mean() for c in set(lab) if c!=lab[i])
        s[i]=(b-a)/max(a,b)
    return s.mean()

def medoid_sil(D,med,lab):
    d=np.sort(D[:,med],axis=1)
    return np.mean(1-d[:,0]/np.where(d[:,1]==0,np.nan,d[:,1]))

for use_sqrt in (True, False):
    D=jsd(P,sqrt=use_sqrt)
    print(f"\n=== 距離 = {'sqrt(JSD)' if use_sqrt else 'JSD (平方根なし)'} ===")
    rows=[]
    for k in range(2,13):
        med,lab=pam(D,k)
        rows.append(dict(k=k, avg_sil=round(sil(D,lab),4), medoid_sil=round(medoid_sil(D,med,lab),4),
                         sizes=sorted(np.bincount(lab),reverse=True)))
    r=pd.DataFrame(rows); print(r.to_string(index=False))
    best_a=int(r.loc[r.avg_sil.idxmax(),"k"]); best_m=int(r.loc[r.medoid_sil.idxmax(),"k"])
    print(f"avg silhouette 最大: k={best_a}   medoid silhouette 最大: k={best_m}")
    if use_sqrt:
        med,lab=pam(D,8)
        pd.DataFrame({"siid":P.index,"cluster":lab}).to_pickle("cst_v2.pkl")
        prof=(P*100).assign(cl=lab).groupby("cl").mean()
        print("\nk=8 のクラスタ内訳:")
        for c in range(8):
            top=prof.loc[c].sort_values(ascending=False).head(3)
            print(f"  cluster {c}: n={int((lab==c).sum()):3d}  "+", ".join(f"{i} {v:.1f}%" for i,v in top.items()))
