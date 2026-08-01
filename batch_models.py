import pandas as pd, numpy as np, warnings, itertools
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

meta = pd.read_pickle("meta.pkl"); g = pd.read_pickle("genus.pkl"); f = pd.read_pickle("family.pkl")
g = g.set_index("siid"); f = f.set_index("siid")
gcols = [c for c in g.columns if c not in ("Sample","total_all")]

# ---- 分母の同定 ----
print("=== 分母の同定（原稿 Summed Entero pre中央値 44.64 / un中央値 0.20 に一致するのはどれか）===")
def build(den):
    rel = pd.DataFrame(index=g.index)
    for c in ["Klebsiella","Escherichia","Enterobacter","Serratia"]:
        rel[c] = g[c]/den*100
    rel["Summed_Entero"] = f["Enterobacteriaceae"]/den*100
    rel["unclassified_Entero"] = rel["Summed_Entero"] - rel[["Klebsiella","Escherichia","Enterobacter","Serratia"]].sum(axis=1)
    return rel

dens = {
 "A) 全genus（Unclassified hit 含む）": g[gcols].sum(axis=1),
 "B) Unclassified hit 除外":            g[[c for c in gcols if c!="Unclassified hit"]].sum(axis=1),
 "C) B かつ Not determined も除外":     g[[c for c in gcols if c not in ("Unclassified hit","Not determined")]].sum(axis=1),
}
for name, den in dens.items():
    rel = build(den)
    dd = meta.merge(rel.reset_index(), on="siid")
    dd["pre"] = ((dd.affected==1)&(dd.dol<dd.ae_dol))
    dd = dd[~((dd.affected==1)&(~dd.pre))]
    a = dd.loc[dd.pre,"Summed_Entero"]; b = dd.loc[~dd.pre,"Summed_Entero"]
    print(f"{name:38s} pre中央値 {a.median():6.2f}  un中央値 {b.median():5.2f}")

# 採用: A（原稿値に最も近いもの）
den = dens["A) 全genus（Unclassified hit 含む）"]
rel = build(den)
d = meta.merge(rel.reset_index(), on="siid")
d["pre"] = ((d.affected==1)&(d.dol<d.ae_dol))
d = d[~((d.affected==1)&(~d.pre))].copy()
d["y"] = d["pre"].astype(int)
d["batch"] = d["batch"].astype(int)
d["logK"] = np.log10(d["Klebsiella"] + 0.01)
d["logE"] = np.log10(d["Summed_Entero"] + 0.01)

print("\n" + "="*70)
print("【1】バッチ構造の診断")
print("="*70)
print("\n-- バッチ×施設 --")
ct = pd.crosstab(d.batch, d.site, margins=True); print(ct)
chi = stats.chi2_contingency(pd.crosstab(d.batch, d.site))
print(f"χ² 検定: p = {chi[1]:.3g}  （バッチと施設は独立でない）")
print("\n-- バッチ×群（pre-onset / unaffected）--")
print(pd.crosstab(d.batch, np.where(d.y==1,"pre-onset","unaffected"), margins=True))
print("\n-- バッチ×機種（原稿の2水準変数）--")
print(pd.crosstab(d.batch, d.instrument))
print("\n-- 症例のバッチ帰属（症例＝クラスタ単位）--")
nb = d.groupby("case")["batch"].nunique()
print(f"単一バッチの症例: {(nb==1).sum()} / {len(nb)}   複数バッチ: {(nb>1).sum()}")
print("→ バッチはほぼ症例レベルの変数（症例がバッチに入れ子）")

print("\n" + "="*70)
print("【2】Summed Enterobacteriaceae: 調整モデルの比較")
print("="*70)
def gee_p(formula, data, groups):
    m = smf.gee(formula, groups=groups, data=data, cov_struct=sm.cov_struct.Exchangeable()).fit()
    return m
def lmm_p(formula, data, groups):
    return smf.mixedlm(formula, data, groups=groups).fit(reml=False)

results = []
specs = [
 ("調整なし",                          "Summed_Entero ~ y"),
 ("機種2水準で調整（原稿）",           "Summed_Entero ~ y + C(instrument)"),
 ("バッチ5水準 固定効果",              "Summed_Entero ~ y + C(batch)"),
 ("バッチ5水準 + 施設",                "Summed_Entero ~ y + C(batch) + site"),
 ("バッチ5水準 + 在胎週数 + DOL",      "Summed_Entero ~ y + C(batch) + ga + dol"),
]
for label, fml in specs:
    try:
        ml = lmm_p(fml, d, d["case"]); lp = ml.pvalues.get("y", np.nan); lc = ml.params.get("y", np.nan)
    except Exception as e:
        lp, lc = np.nan, np.nan
    try:
        mg = gee_p(fml, d, d["case"]); gp = mg.pvalues.get("y", np.nan); gc = mg.params.get("y", np.nan)
    except Exception as e:
        gp, gc = np.nan, np.nan
    results.append(dict(model=label, LMM_coef=round(lc,1), LMM_p=f"{lp:.4f}",
                        GEE_coef=round(gc,1), GEE_p=f"{gp:.4f}"))
print(pd.DataFrame(results).to_string(index=False))
print("（係数 = pre-onset群の相対量差, %ポイント。infant をクラスタ／ランダム切片）")

print("\n-- バッチ5水準ランダム切片（症例はバッチに入れ子のため交差なし）--")
try:
    m = smf.mixedlm("Summed_Entero ~ y", d, groups=d["batch"], re_formula="1").fit(reml=False)
    print(f"バッチをランダム切片:  y 係数 {m.params['y']:.1f},  p = {m.pvalues['y']:.4f}")
except Exception as e:
    print("収束せず:", e)

print("\n" + "="*70)
print("【3】バッチ別（層別）解析")
print("="*70)
def cliffs(a,b):
    a=np.asarray(a); b=np.asarray(b)
    return ((a[:,None]>b[None,:]).sum()-(a[:,None]<b[None,:]).sum())/(len(a)*len(b))
rows=[]
for bt in sorted(d.batch.unique()):
    s = d[d.batch==bt]
    a = s.loc[s.y==1,"Summed_Entero"]; b = s.loc[s.y==0,"Summed_Entero"]
    rows.append(dict(batch=bt, n_pre=len(a), n_un=len(b),
        pre_med=None if len(a)==0 else round(a.median(),2), un_med=round(b.median(),2),
        cliff=None if len(a)==0 else round(cliffs(a,b),2),
        MW_p=None if len(a)==0 else round(stats.mannwhitneyu(a,b,alternative="greater").pvalue,4)))
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "="*70)
print("【4】Site B 限定 GEE（Klebsiella → 合併症）: 単位を明示して再計算")
print("="*70)
db = d[d.site=="B"].copy()
print(f"Site B: 検体 {len(db)}（pre-onset {db.y.sum()}, unaffected {(db.y==0).sum()}）, 症例 {db.case.nunique()}")
for label, var, scale in [("相対量 1%あたり","Klebsiella",1.0),
                          ("相対量 10%あたり","Klebsiella",10.0),
                          ("1SDあたり","Klebsiella", db["Klebsiella"].std()),
                          ("log10(K+0.01) 1単位あたり","logK",1.0)]:
    db["x"] = db[var]/scale
    for extra, nm in [("", "調整なし"), (" + C(batch)", "バッチ5水準調整")]:
        try:
            m = smf.gee("y ~ x"+extra, groups=db["case"], data=db,
                        family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
            or_ = np.exp(m.params["x"]); ci = np.exp(m.conf_int().loc["x"])
            print(f"{label:26s} {nm:14s} OR {or_:8.2f}  95%CI {ci[0]:.2f}-{ci[1]:.2f}  p={m.pvalues['x']:.4f}")
        except Exception as e:
            print(f"{label:26s} {nm:14s} 収束せず ({type(e).__name__})")

print("\n" + "="*70)
print("【5】患者単位 完全置換検定（バッチ層内置換との比較）")
print("="*70)
inf = d.groupby("case").agg(site=("site","first"), batch=("batch","first"),
                            aff=("y","max")).reset_index()
def stat_of(aff_cases, data, var="Summed_Entero"):
    m = data.case.isin(aff_cases)
    return data.loc[m,var].mean() - data.loc[~m,var].mean()
obs_cases = set(inf.loc[inf.aff==1,"case"])
for scope, pool in [("全43症例から5例を選ぶ", inf),
                    ("Site B の24症例から5例", inf[inf.site=="B"]),
                    ("バッチ内層別（各バッチ内で発症例数を保持）", None)]:
    if pool is not None:
        sub = d[d.case.isin(pool.case)]
        obs = stat_of(obs_cases, sub)
        cnt = tot = 0
        for comb in itertools.combinations(sorted(pool.case), 5):
            tot += 1
            if abs(stat_of(set(comb), sub)) >= abs(obs) - 1e-9: cnt += 1
        print(f"{scope:42s} 列挙 {tot:>8,d}  両側P = {cnt/tot:.4f}")
    else:
        # バッチ内層別置換（発症例が属するバッチ内でのみ入れ替え）
        bcount = inf.loc[inf.aff==1].groupby("batch").size().to_dict()
        pools = {b: sorted(inf.loc[inf.batch==b,"case"]) for b in bcount}
        obs = stat_of(obs_cases, d)
        allsets = []
        for b, k in bcount.items():
            allsets.append(list(itertools.combinations(pools[b], k)))
        cnt = tot = 0
        for prod in itertools.product(*allsets):
            sel = set(itertools.chain.from_iterable(prod)); tot += 1
            if abs(stat_of(sel, d)) >= abs(obs) - 1e-9: cnt += 1
        print(f"{scope:42s} 列挙 {tot:>8,d}  両側P = {cnt/tot:.4f}")
        print("  （発症例のバッチ内訳: " + ", ".join(f"batch{b}:{k}例" for b,k in sorted(bcount.items())) + "）")

d.to_pickle("d_final.pkl")
