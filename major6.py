import pandas as pd, numpy as np, itertools, warnings
import statsmodels.api as sm, statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
d=pd.read_pickle("d_final.pkl"); d["batch"]=d["batch"].astype(int)
db=d[d.site=="B"].copy()
print(f"Site B: {len(db)}検体 (pre-onset {int(db.y.sum())}, unaffected {int((db.y==0).sum())}), 症例 {db.case.nunique()}")

print("\n【A】相対量を従属変数とするモデル（構造的に適切な向き）")
for lab,dat in [("Site B",db),("全体",d)]:
    for var in ["Klebsiella","Summed_Entero"]:
        lm=smf.mixedlm(f"{var} ~ y", dat, groups=dat["case"]).fit(reml=False)
        ge=smf.gee(f"{var} ~ y", groups=dat["case"], data=dat,
                   cov_struct=sm.cov_struct.Exchangeable()).fit()
        print(f"  {lab:5s} {var:14s} 差 {lm.params['y']:6.1f}%p  LMM P={lm.pvalues['y']:.4f}   GEE P={ge.pvalues['y']:.4f}")

print("\n【B】参考: ロジスティック OR（従属変数=発症前, 交換可能作業相関）")
db["Klog"]=np.log10(db.Klebsiella+0.01)
for lab,x in [("10倍増あたり log10(K+0.01)","Klog"),("相対量 1%あたり","Klebsiella")]:
    m=smf.gee(f"y ~ {x}", groups=db["case"], data=db, family=sm.families.Binomial(),
              cov_struct=sm.cov_struct.Exchangeable()).fit()
    ci=np.exp(m.conf_int().loc[x])
    print(f"  {lab:28s} OR {np.exp(m.params[x]):.2f} (95%CI {ci[0]:.2f}-{ci[1]:.2f}) P={m.pvalues[x]:.4f}")

print("\n【C】患者単位 完全置換検定（統計量=発症割付群の検体平均 − 非割付群の検体平均）")
def perm(data,var,strat=None,restrict=None):
    if restrict is not None: data=data[data.case.isin(restrict)]
    cases=sorted(data.case.unique()); idx={c:i for i,c in enumerate(cases)}
    s=np.zeros(len(cases)); n=np.zeros(len(cases))
    for c,gg in data.groupby("case"): s[idx[c]]=gg[var].sum(); n[idx[c]]=len(gg)
    S,N=s.sum(),n.sum()
    aff=sorted(set(data.loc[data.y==1,"case"]))
    st=lambda sel:(s[list(sel)].sum()/n[list(sel)].sum())-((S-s[list(sel)].sum())/(N-n[list(sel)].sum()))
    obs=st([idx[c] for c in aff])
    if strat is None:
        combs=itertools.combinations(range(len(cases)),len(aff))
    else:
        cb=data.groupby("case")[strat].agg(lambda x:x.mode().min())
        layers={b:[idx[c] for c in cb[cb==b].index] for b in sorted(cb.unique())}
        need={b:sum(1 for c in aff if cb[c]==b) for b in layers}
        need={b:k for b,k in need.items() if k>0}
        parts=[list(itertools.combinations(layers[b],k)) for b,k in sorted(need.items())]
        combs=(tuple(itertools.chain.from_iterable(p)) for p in itertools.product(*parts))
    vals=[st(c) for c in combs]
    v=np.array(vals)
    return obs, len(v), (np.abs(v)>=abs(obs)-1e-9).mean()
for var in ["Klebsiella","Summed_Entero"]:
    o,n1,p1=perm(db,var)
    keep=set(db.groupby("case")["batch"].agg(lambda x:x.mode().min()).pipe(
        lambda cb: cb[cb.isin(db.loc[db.y==1].groupby("case")["batch"].agg(lambda x:x.mode().min()).unique())].index))
    o2,n2,p2=perm(db,var,strat="batch",restrict=keep)
    print(f"  {var:14s} SiteB 全列挙 {n1:>6,d} 通り P={p1:.4f} | SiteB×バッチ層内 {n2:>6,d} 通り P={p2:.4f}")
