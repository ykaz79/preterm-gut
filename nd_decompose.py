import pandas as pd, numpy as np
g=pd.read_pickle("genus.pkl").set_index("siid"); f=pd.read_pickle("family.pkl").set_index("siid")
gc=[c for c in g.columns if c not in ("Sample","total_all","Not determined","Unclassified hit")]
fc=[c for c in f.columns if c not in ("Sample","Not determined","Unclassified hit")]
G=g[gc].astype(int); F=f[fc].astype(int)

# 属→科 の対応を「全検体で属カウント ≤ 科カウント」から導出
assign={}; amb={}
for gg in gc:
    cand=[ff for ff in fc if (G[gg].values<=F[ff].values).all()]
    (assign if len(cand)==1 else amb)[gg]=cand[0] if len(cand)==1 else cand
print("一意に決まった属:",len(assign)," 曖昧:",len(amb))

for _ in range(10):
    if not amb: break
    resid = F.copy()
    for gg,ff in assign.items(): resid[ff]-=G[gg]
    newly={}
    for gg,cands in list(amb.items()):
        ok=[ff for ff in cands if (G[gg].values<=resid[ff].values).all()]
        if len(ok)==1: newly[gg]=ok[0]; del amb[gg]
        elif len(ok)==0: newly[gg]=None; del amb[gg]
        else: amb[gg]=ok
    if not newly: break
    assign.update({k:v for k,v in newly.items() if v})
print("最終: 割当済",len(assign)," 未解決",len(amb))
if amb: print("  未解決例:", list(amb.items())[:5])

resid = F.copy()
for gg,ff in assign.items(): resid[ff]-=G[gg]
print("\n残差(=Unclassified <Family>)の負値:", int((resid.values<0).sum()))
nd = g["Not determined"].astype(int)
print("残差合計 == 'Not determined' ?", bool((resid.sum(axis=1).values==nd.values).all()),
      " 最大差:", int((resid.sum(axis=1)-nd).abs().max()))
tot=g[[c for c in g.columns if c not in ('Sample','total_all')]].sum(axis=1)
print("\n=== 分解後 Unclassified <Family> の平均相対量 上位 ===")
print((resid.div(tot,axis=0)*100).mean().sort_values(ascending=False).head(10).round(3).to_string())
resid.columns=["Unclassified "+c for c in resid.columns]
X = pd.concat([G, resid], axis=1)
X.to_pickle("feature_table.pkl")
print("\n分解後の特徴量テーブル:", X.shape)
