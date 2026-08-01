import pandas as pd, numpy as np, itertools
d=pd.read_pickle("d_final.pkl")
g=pd.read_pickle("genus.pkl").set_index("siid"); f=pd.read_pickle("family.pkl").set_index("siid")
den=g[[c for c in g.columns if c not in ("Sample","total_all")]].sum(axis=1)
d=d.copy()
d["Enterobacter"]=d.siid.map(g["Enterobacter"]/den*100)
d["Serratia"]=d.siid.map(g["Serratia"]/den*100)
d["Escherichia"]=d.siid.map(g["Escherichia"]/den*100)

cases=sorted(d.case.unique()); idx={c:i for i,c in enumerate(cases)}
aff=[idx[c] for c in sorted(d.loc[d.y==1,"case"].unique())]
combs=np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(len(cases)),5)),
                  dtype=np.int32).reshape(-1,5)
print("列挙数:", len(combs))
n=np.array([len(d[d.case==c]) for c in cases], float)
N=n.sum()
rows=[]
for tax,col in [("Summed Enterobacteriaceae","Summed_Entero"),("Klebsiella","Klebsiella"),
                ("Escherichia","Escherichia"),("unclassified Enterobacteriaceae","unclassified_Entero"),
                ("Enterobacter","Enterobacter"),("Serratia","Serratia")]:
    s=np.array([d.loc[d.case==c,col].sum() for c in cases], float); S=s.sum()
    A=s[combs].sum(1); M=n[combs].sum(1)
    stat=A/M-(S-A)/(N-M)
    obs=(s[aff].sum()/n[aff].sum())-((S-s[aff].sum())/(N-n[aff].sum()))
    p=(np.abs(stat)>=abs(obs)-1e-9).mean()
    rows.append(dict(Taxon=tax, 観測差=round(obs,2),
                     新定義P=("<0.0001" if p<1e-4 else f"{p:.4f}"),
                     原稿記載P={"Summed Enterobacteriaceae":0.008,"Klebsiella":0.005,"Escherichia":0.121,
                               "unclassified Enterobacteriaceae":0.685,"Enterobacter":0.685,"Serratia":0.685}.get(tax,"")))
print(pd.DataFrame(rows).to_string(index=False))
