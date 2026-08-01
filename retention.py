import pandas as pd, numpy as np
from scipy import stats

cst = pd.read_pickle("cst.pkl"); meta = pd.read_pickle("meta.pkl"); g = pd.read_pickle("genus.pkl")
full = meta[meta.siid.isin(set(g.siid))].merge(cst, on="siid")
full["C2"] = (full.cluster==1).astype(int)
full["pre"]  = ((full.affected==1)&(full.dol<full.ae_dol))
full["post"] = ((full.affected==1)&(full.dol>=full.ae_dol))
full = full.sort_values(["case","t"])

def pairs(df, mode):
    """mode='adjacent': 隣接時点(T_i→T_{i+1})の両方が解析可能な組のみ
       mode='consecutive': 欠測時点を飛ばした連続する解析可能検体の組"""
    out=[]
    for case, s in df.groupby("case"):
        s = s.sort_values("t")
        ts = s.t.tolist(); cs = s.C2.tolist(); cl = s.cluster.tolist()
        for i in range(len(s)-1):
            if mode=="adjacent" and ts[i+1]-ts[i] != 1: continue
            out.append(dict(case=case, t_from=ts[i], t_to=ts[i+1],
                            from_C2=cs[i], to_C2=cs[i+1],
                            from_cl=cl[i], to_cl=cl[i+1],
                            affected=s.affected.iloc[0]))
    return pd.DataFrame(out)

def summarise(title, df, mode):
    P = pairs(df, mode)
    print(f"\n{'='*66}\n{title}（{ '隣接時点のみ' if mode=='adjacent' else '欠測を飛ばした連続検体' }）\n{'='*66}")
    print(f"全遷移ペア数: {len(P)}")
    tot = P[P.from_C2==1]
    print(f"C2 から出発するペア（分母）: {len(tot)}   うち C2 に留まる: {int(tot.to_C2.sum())}"
          f"  → 自己保持率 {tot.to_C2.mean()*100:.1f}%")
    rows=[]
    for lab, sub in [("合併症児", tot[tot.affected==1]), ("非合併症児", tot[tot.affected==0])]:
        n=len(sub); k=int(sub.to_C2.sum())
        rows.append(dict(群=lab, 分母_C2発ペア=n, C2維持=k, 離脱=n-k,
                         自己保持率=f"{k/n*100:.1f}%" if n else "—"))
    r=pd.DataFrame(rows); print(); print(r.to_string(index=False))
    a=tot[tot.affected==1]; b=tot[tot.affected==0]
    if len(a) and len(b):
        ft=stats.fisher_exact([[int(a.to_C2.sum()), len(a)-int(a.to_C2.sum())],
                               [int(b.to_C2.sum()), len(b)-int(b.to_C2.sum())]])
        print(f"Fisher 正確検定（群間の自己保持率の差）: P = {ft.pvalue:.4f}")
    # 参考: 全型の自己保持率
    allp = (P.assign(same=(P["from_cl"]==P["to_cl"]).astype(int))
             .groupby("from_cl")
             .agg(分母=("same","size"), 自己保持=("same","sum")))
    allp["率"] = (allp["自己保持"]/allp["分母"]*100).round(1).astype(str)+"%"
    return P, allp

NAME={7:"C1",1:"C2",4:"C3",2:"C4",0:"C5",5:"C6",6:"C7",3:"C8"}

P1,a1 = summarise("A) 全206検体（発症後検体を含む・原稿 Fig 2 と同じ範囲）", full, "adjacent")
print("\n参考: 全型の自己保持率（隣接時点）")
a1.index=[NAME[i] for i in a1.index]; print(a1.sort_index().to_string())

P2,_ = summarise("B) 全206検体・欠測を飛ばした連続検体", full, "consecutive")

ana = full[~full.post]
P3,_ = summarise("C) 発症後検体を除外（197検体）", ana, "adjacent")

print("\n=== 合併症児の C2 発ペア 明細 ===")
tot = P1[(P1.from_C2==1)&(P1.affected==1)]
tot = tot.assign(遷移=[f"{NAME[a]}→{NAME[b]}" for a,b in zip(tot.from_cl,tot.to_cl)])
print(tot[["case","t_from","t_to","遷移"]].to_string(index=False))
print("\n=== 非合併症児の C2 発ペア 明細 ===")
tot = P1[(P1.from_C2==1)&(P1.affected==0)]
tot = tot.assign(遷移=[f"{NAME[a]}→{NAME[b]}" for a,b in zip(tot.from_cl,tot.to_cl)])
print(tot[["case","t_from","t_to","遷移"]].to_string(index=False))
