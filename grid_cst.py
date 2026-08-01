import pandas as pd, numpy as np, itertools
X0=pd.read_pickle("feature_table.pkl"); g=pd.read_pickle("genus.pkl").set_index("siid")
allg=[c for c in g.columns if c not in ("Sample","total_all")]
den_all=g[allg].sum(axis=1)
den_noUH=g[[c for c in allg if c!="Unclassified hit"]].sum(axis=1)
TARGET=sorted([37,42,24,33,44,10,7,9],reverse=True)

def jsd(P,sq):
    A=P.values;n=len(A);D=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            p,q=A[i],A[j];m=0.5*(p+q);a=p>0;b=q>0
            d=0.5*np.sum(p[a]*np.log2(p[a]/m[a]))+0.5*np.sum(q[b]*np.log2(q[b]/m[b]))
            d=max(d,0.); D[i,j]=D[j,i]=np.sqrt(d) if sq else d
    return D
def pam(D,k):
    n=len(D);med=[int(np.argmin(D.sum(1)))]
    while len(med)<k:
        cur=D[:,med].min(1);best,bg=None,-np.inf
        for c in range(n):
            if c in med:continue
            gn=np.maximum(cur-D[:,c],0).sum()
            if gn>bg:bg,best=gn,c
        med.append(best)
    imp=True
    while imp:
        imp=False;cost=D[:,med].min(1).sum()
        for mi in range(k):
            for h in range(n):
                if h in med:continue
                tr=med.copy();tr[mi]=h;c=D[:,tr].min(1).sum()
                if c<cost-1e-12:med,cost,imp=tr,c,True
    return np.array(med),np.argmin(D[:,med],1)

res=[]
for den_name,den in [("全リード",den_all),("UH除外",den_noUH)]:
  for thr_first in [True,False]:
    for prev_rule in ["gt0","ge0.01"]:
      rel=X0.div(den,axis=0)*100
      if thr_first: rel=rel.where(rel>=0.01,0.0)
      keep=[c for c in rel.columns if rel[c].mean()>=0.01 and
            ((rel[c]>0).mean() if prev_rule=="gt0" else (rel[c]>=0.01).mean())>=0.10]
      P=rel[keep]
      if not thr_first: P=P.where(P>=0.01,0.0)
      P=P.div(P.sum(axis=1),axis=0).fillna(0)
      for sq in [True,False]:
        D=jsd(P,sq); med,lab=pam(D,8); sizes=sorted(np.bincount(lab),reverse=True)
        res.append(dict(分母=den_name, 閾値=("先"if thr_first else"後"), prevalence=prev_rule,
                        距離=("sqrt"if sq else"raw"), 特徴量=len(keep), sizes=str(sizes),
                        一致=(sizes==TARGET), 差=sum(abs(a-b) for a,b in zip(sizes,TARGET))))
r=pd.DataFrame(res).sort_values("差")
pd.set_option("display.width",200)
print("原稿のクラスタサイズ:", TARGET)
print(r.to_string(index=False))
