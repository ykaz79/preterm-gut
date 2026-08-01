import pandas as pd, numpy as np
from scipy import stats
meta = pd.read_pickle("meta.pkl"); g = pd.read_pickle("genus.pkl")
d = pd.read_pickle("d_final.pkl")
ok = set(g["siid"])
m206 = meta[meta.siid.isin(ok)].copy(); m206["batch"]=m206["batch"].astype(int)
d = d.copy(); d["batch"]=d["batch"].astype(int)
aff = set(d.loc[d.y==1,"case"])

def show(name, df):
    print("\n--- %s ---" % name)
    print(pd.crosstab(df.batch, df.site, margins=True, margins_name="Total").to_string())
    core = pd.crosstab(df.batch, df.site)
    chi2,p,dof,exp = stats.chi2_contingency(core)
    print("chi2=%.2f, df=%d, P=%.3g   expected<5 cells: %d/%d" % (chi2,dof,p,(exp<5).sum(),exp.size))
    rng = np.random.default_rng(0); obs=core.values
    rows=obs.sum(1); cols=obs.sum(0); cnt=0; B=20000
    rowvec=np.repeat(np.arange(len(rows)),rows); colvec=np.repeat(np.arange(len(cols)),cols)
    for _ in range(B):
        sim=np.zeros_like(obs)
        np.add.at(sim,(rng.permutation(rowvec),colvec),1)
        c2,_,_,_=stats.chi2_contingency(sim)
        if c2>=chi2-1e-9: cnt+=1
    print("Monte Carlo test (B=20,000): P = %.4f" % (cnt/B))

show("A) all analysable samples (n=206)", m206)
show("B) samples used in the comparison (n=197)", d)

cb = d.groupby("case").agg(batch=("batch", lambda s: s.mode().min()), site=("site","first"))
cb["affected"]=[c in aff for c in cb.index]
print("\n--- C) infant level (n=43) ---")
print(pd.crosstab(cb.batch, cb.site, margins=True, margins_name="Total").to_string())
chi2,p,dof,exp = stats.chi2_contingency(pd.crosstab(cb.batch, cb.site))
print("chi2=%.2f, df=%d, P=%.3g" % (chi2,dof,p))

rows=[]
sm=m206.groupby("batch").size(); sa=d.groupby("batch").size(); pre=d[d.y==1].groupby("batch").size()
for b in sorted(cb.batch.unique()):
    s=cb[cb.batch==b]
    rows.append(dict(Batch="B%d"%b, Infants=len(s), A=(s.site=="A").sum(), B=(s.site=="B").sum(),
        Affected=int(s.affected.sum()), Samp206=int(sm.get(b,0)), Samp197=int(sa.get(b,0)),
        Pre=int(pre.get(b,0))))
t=pd.DataFrame(rows); tot=t.iloc[:,1:].sum(); tot["Batch"]="Total"
t=pd.concat([t,pd.DataFrame([tot])[t.columns]],ignore_index=True)
print("\n--- D) combined ---"); print(t.to_string(index=False))

print("\n--- E) collapsed to 2-level instrument (current manuscript variable) ---")
d2=d.copy(); d2["instr"]=np.where(d2.siid<=113,"MiSeq","MiSeq i100")
print(pd.crosstab(d2.instr, d2.site, margins=True, margins_name="Total").to_string())
chi2,p,dof,exp=stats.chi2_contingency(pd.crosstab(d2.instr,d2.site))
print("chi2=%.2f, df=%d, P=%.4f" % (chi2,dof,p))
