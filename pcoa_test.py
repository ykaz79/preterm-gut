import pandas as pd, numpy as np
X0=pd.read_pickle("feature_table.pkl"); g=pd.read_pickle("genus.pkl").set_index("siid")
den=g[[c for c in g.columns if c not in ("Sample","total_all")]].sum(axis=1)
rel=X0.div(den,axis=0)*100; rel=rel.where(rel>=0.01,0.0)
keep=[c for c in rel.columns if rel[c].mean()>=0.01 and (rel[c]>0).mean()>=0.10]
P=rel[keep]; P=P.div(P.sum(axis=1),axis=0)

def jsd(P,sq):
    A=P.values;n=len(A);D=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            p,q=A[i],A[j];m=0.5*(p+q);a=p>0;b=q>0
            d=0.5*np.sum(p[a]*np.log2(p[a]/m[a]))+0.5*np.sum(q[b]*np.log2(q[b]/m[b]))
            d=max(d,0.);D[i,j]=D[j,i]=np.sqrt(d) if sq else d
    return D

def pcoa(D):
    n=len(D); J=np.eye(n)-np.ones((n,n))/n
    B=-0.5*J@(D**2)@J
    w,v=np.linalg.eigh(B); idx=np.argsort(w)[::-1]; w=w[idx]
    pos=w[w>0]
    return w, pos/pos.sum()*100, w[w<0].sum()/np.abs(w).sum()*100

def triangle_violations(D, n_check=200000, seed=0):
    rng=np.random.default_rng(seed); n=len(D); bad=0
    for _ in range(n_check):
        i,j,k=rng.integers(0,n,3)
        if len({i,j,k})<3: continue
        if D[i,j] > D[i,k]+D[k,j]+1e-12: bad+=1
    return bad/n_check*100

for lab,sq in [("sqrt(JSD)  = Jensen–Shannon 距離", True), ("JSD        = 平方根なし", False)]:
    D=jsd(P,sq); w,pct,negshare=pcoa(D)
    print(f"\n=== {lab} ===")
    print("  PCo1-PCo5 寄与率: " + ", ".join(f"{p:.1f}%" for p in pct[:5]))
    print(f"  負の固有値が占める割合: {negshare:.2f}%   最小固有値 {w.min():.4f}")
    print(f"  三角不等式の違反率（20万組の無作為抽出）: {triangle_violations(D):.2f}%")
print("\n原稿 Fig.1a の表示: PCo1 24.6%, PCo2 16.4%")
