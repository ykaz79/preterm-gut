import pandas as pd, numpy as np, itertools

d = pd.read_pickle("d_final.pkl")

def exact_perm(data, var, k=5, strat=None):
    """症例単位の完全置換。統計量 = 選択症例の検体平均 − 非選択症例の検体平均"""
    cases = sorted(data.case.unique())
    idx = {c:i for i,c in enumerate(cases)}
    s = np.zeros(len(cases)); n = np.zeros(len(cases))
    for c, gg in data.groupby("case"):
        s[idx[c]] = gg[var].sum(); n[idx[c]] = len(gg)
    S, N = s.sum(), n.sum()
    obs_cases = sorted(data.loc[data.y==1,"case"].unique())
    def stat(sel):
        a = s[sel].sum(); m = n[sel].sum()
        if m == 0 or N-m == 0: return np.nan
        return a/m - (S-a)/(N-m)
    obs = stat([idx[c] for c in obs_cases])

    if strat is None:
        combs = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(len(cases)), k)),
                            dtype=np.int32).reshape(-1, k)
    else:
        # 層（バッチ）内で発症例数を保持した全組合せ
        layers = {}
        for c in cases:
            layers.setdefault(data.loc[data.case==c, strat].iloc[0], []).append(idx[c])
        need = data.loc[data.y==1].groupby(strat)["case"].nunique().to_dict()
        parts = [list(itertools.combinations(layers[b], kk)) for b, kk in sorted(need.items())]
        combs = np.array([list(itertools.chain.from_iterable(p)) for p in itertools.product(*parts)],
                         dtype=np.int32)
    A = s[combs].sum(axis=1); M = n[combs].sum(axis=1)
    stats_ = A/M - (S-A)/(N-M)
    p = (np.abs(stats_) >= abs(obs) - 1e-9).mean()
    return obs, len(combs), p

print("=== 患者単位 完全置換検定 ===\n")
for var in ["Summed_Entero","Klebsiella"]:
    obs, nc, p = exact_perm(d, var)
    print(f"{var:15s} 全43症例から5例  列挙 {nc:>9,d}  観測差 {obs:6.2f}%  両側P = {p:.4f}")
db = d[d.site=="B"]
for var in ["Summed_Entero","Klebsiella"]:
    obs, nc, p = exact_perm(db, var)
    print(f"{var:15s} SiteB 24症例から5例 列挙 {nc:>9,d}  観測差 {obs:6.2f}%  両側P = {p:.4f}")
print()
for var in ["Summed_Entero","Klebsiella"]:
    obs, nc, p = exact_perm(d, var, strat="batch")
    print(f"{var:15s} バッチ層内置換     列挙 {nc:>9,d}  観測差 {obs:6.2f}%  両側P = {p:.4f}")
for var in ["Summed_Entero","Klebsiella"]:
    obs, nc, p = exact_perm(db, var, strat="batch")
    print(f"{var:15s} SiteB×バッチ層内   列挙 {nc:>9,d}  観測差 {obs:6.2f}%  両側P = {p:.4f}")

print("\n=== 発症例のバッチ帰属 ===")
aff = d[d.y==1].groupby("case").agg(batch=("batch","first"), site=("site","first"),
                                    n=("siid","size"), onset=("ae_dol","first"))
print(aff)
print("\n各バッチの症例数（SiteB）:")
print(db.groupby("batch")["case"].nunique())
