import pandas as pd, numpy as np
from scipy import stats
X0=pd.read_pickle("feature_table.pkl"); g=pd.read_pickle("genus.pkl").set_index("siid")
meta=pd.read_pickle("meta.pkl"); cst=pd.read_pickle("cst_v2.pkl")
den=g[[c for c in g.columns if c not in ("Sample","total_all")]].sum(axis=1)
NAME={7:"C1",1:"C2",4:"C3",2:"C4",0:"C5",5:"C6",6:"C7",3:"C8"}

m=meta[meta.siid.isin(set(g.index))].merge(cst,on="siid"); m["PGCT"]=m.cluster.map(NAME)
B=m[m.site=="B"].copy()
sid=B.siid.values
print(f"Site B 解析済検体: {len(sid)}  児 {B.case.nunique()}（発症5）")

# --- 全workflowを Site B 内で再実施 ---
rel=X0.loc[sid].div(den.loc[sid],axis=0)*100
rel=rel.where(rel>=0.01,0.0)
keep=[c for c in rel.columns if rel[c].mean()>=0.01 and (rel[c]>0).mean()>=0.10]
print(f"Site B 内で保持された特徴量: {len(keep)}（全体解析では 21）")
print("  全体のみ:", sorted(set(pd.read_pickle('feature_table.pkl').columns) & set()) or "")
P=rel[keep]; P=P.div(P.sum(axis=1),axis=0)

def jsd(P):
    A=P.values;n=len(A);D=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            p,q=A[i],A[j];mm=0.5*(p+q);a=p>0;b=q>0
            d=0.5*np.sum(p[a]*np.log2(p[a]/mm[a]))+0.5*np.sum(q[b]*np.log2(q[b]/mm[b]))
            D[i,j]=D[j,i]=np.sqrt(max(d,0.))
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
def sil(D,lab):
    n=len(D);s=np.zeros(n)
    for i in range(n):
        sm=(lab==lab[i]).copy();sm[i]=False
        if sm.sum()==0:continue
        a=D[i,sm].mean();b=min(D[i,lab==c].mean() for c in set(lab) if c!=lab[i])
        s[i]=(b-a)/max(a,b)
    return s.mean()

D=jsd(P)
print("\n=== k の探索（Site B 内, k=3-12）===")
res={}
for k in range(3,13):
    med,lab=pam(D,k); res[k]=(sil(D,lab),lab)
    print(f"  k={k:2d}  平均シルエット幅 {res[k][0]:.4f}  サイズ {sorted(np.bincount(lab),reverse=True)}")
bestk=max(res,key=lambda k:res[k][0]); print(f"→ 最適 k = {bestk}")

for k in sorted({bestk,8}):
    lab=res[k][1]
    print(f"\n=== Site B 内 k={k} のクラスタ プロファイル ===")
    prof=(P*100).assign(cl=lab).groupby("cl").mean()
    kls=[]
    for c in range(k):
        top=prof.loc[c].sort_values(ascending=False).head(3)
        tag=""
        if top.index[0] in ("Klebsiella","Unclassified Enterobacteriaceae"): kls.append(c); tag="  ← Klebsiella/Enterobacteriaceae 優位"
        print(f"  B{c+1}: n={int((lab==c).sum()):3d}  "+", ".join(f"{i} {v:.1f}%" for i,v in top.items())+tag)
    print("  Klebsiella優位クラスタ:", [f"B{c+1}" for c in kls])
    if kls:
        B["newcl"]=lab; B["inKl"]=B.newcl.isin(kls).astype(int)
        print(f"\n  --- 全体解析の C2 との対応（k={k}）---")
        print(pd.crosstab(B.PGCT, np.where(B.inKl==1,"Kleb-dominant","other")).to_string())
        sub=B[~((B.affected==1)&(B.dol>=B.ae_dol))].copy()
        sub["pre"]=((sub.affected==1)&(sub.dol<sub.ae_dol)).astype(int)
        sub["inKl"]=sub.newcl.isin(kls).astype(int)
        ct=pd.crosstab(sub.inKl,sub.pre)
        print(f"\n  --- 発症前検体の濃縮（Site B 内, 107検体）---")
        print(ct.to_string())
        print("  Fisher P =", round(stats.fisher_exact(ct.values).pvalue,4))
        ev=sub.groupby("case").agg(y=("pre","max"), ever=("inKl","max"))
        c=pd.crosstab(ev.ever,ev.y)
        print(f"\n  --- 児レベル 2x2（Site B 24児）---"); print(c.to_string())
        a=c.loc[1,1] if (1 in c.index and 1 in c.columns) else 0
        b=c.loc[1,0] if (1 in c.index and 0 in c.columns) else 0
        print(f"  PPV = {a}/{a+b} = {a/(a+b)*100:.1f}%   Fisher P = {round(stats.fisher_exact(c.values).pvalue,4)}")
