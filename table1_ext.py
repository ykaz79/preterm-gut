import pandas as pd, numpy as np, re
from openpyxl import load_workbook
wb=load_workbook("/mnt/user-data/uploads/BGM研究ノート学会短縮版_v2.xlsx", data_only=True)
ws=wb["シート1"]; rows=[list(r) for r in ws.iter_rows(values_only=True)]
cases=[str(x).strip() for x in rows[1][1:] if x is not None]; n=len(cases)
F={str(r[0]).strip():list(r[1:1+n]) for r in rows[2:] if r[0] is not None}
def ga(x):
    if isinstance(x,(int,float)): return float(x)
    t=str(x).strip(); m=re.match(r"^(\d+)\s+(\d)/7$",t)
    return int(m.group(1))+int(m.group(2))/7 if m else float(t)
def num(x):
    try: return float(str(x))
    except: return np.nan

D=pd.DataFrame({"case":cases})
D["site"]=F["Site"]; D["ga"]=[ga(v) for v in F["Gestational age"]]
D["sex"]=[num(v) for v in F["Sex(male)"]]
D["cam"]=[num(v) for v in F["CAM"]]
D["ae"]=[num(v) for v in F["AE(DOL)"]]
D["affected"]=D.ae.notna()
for t in range(6): D[f"bwz{t}"]=[num(v) for v in F[f"Bw z-score(T{t})"]]
for t in range(1,6):
    D[f"nut{t}"]=[str(v) for v in F[f"Nutrition(T{t})"]]
    D[f"abx{t}"]=[str(v) for v in F[f"ABx(T{t})"]]
    D[f"dol{t}"]=[num(v) for v in F[f"DOL(T{t})"]]
D["t5_missing"]=[str(F["T5"][i]).strip().upper()=="NA" for i in range(n)]

nut=D[[f"nut{t}" for t in range(1,6)]]; abx=D[[f"abx{t}" for t in range(1,6)]]
D["abx_any"]=(abx.isin(["E","T"])).any(axis=1)
D["abx_E_T1"]=D.abx1=="E"
D["abx_T_any"]=(abx=="T").any(axis=1)
D["abx_n"]=(abx.isin(["E","T"])).sum(axis=1)
D["abx_none"]=~D.abx_any
D["nut_B_T1"]=D.nut1=="B"
D["nut_B_all"]=(nut=="B").all(axis=1)
D["nut_B_n"]=(nut=="B").sum(axis=1)
D["nut_F_any"]=(nut=="F").any(axis=1)
D["nut_NPO_any"]=(nut=="O").any(axis=1)
D["dz"]=D.bwz5-D.bwz0

def med(x): 
    x=pd.Series(x).dropna()
    return f"{x.median():.2f} ({x.quantile(.25):.2f}\u2013{x.quantile(.75):.2f})" if len(x) else "\u2014"
def med1(x):
    x=pd.Series(x).dropna()
    return f"{x.median():.0f} ({x.quantile(.25):.0f}\u2013{x.quantile(.75):.0f})" if len(x) else "\u2014"
def pct(x):
    x=pd.Series(x).dropna().astype(bool)
    return f"{int(x.sum())} ({x.mean()*100:.1f})"

grp=[("Overall (n = 43)",D),("Unaffected (n = 38)",D[~D.affected]),("Affected (n = 5)",D[D.affected])]
rows_out=[]
def add(label, fn):
    rows_out.append([label]+[fn(g) for _,g in grp])
add("Antibiotic exposure at any timepoint, n (%)", lambda g: pct(g.abx_any))
add("  Empirical therapy at T1, n (%)",            lambda g: pct(g.abx_E_T1))
add("  Therapeutic use at any timepoint, n (%)",   lambda g: pct(g.abx_T_any))
add("  No antibiotic exposure, n (%)",             lambda g: pct(g.abx_none))
add("  Timepoints with antibiotic exposure, n",    lambda g: med1(g.abx_n))
add("Breast milk >80% at T1, n (%)",               lambda g: pct(g.nut_B_T1))
add("Breast milk >80% at all timepoints, n (%)",   lambda g: pct(g.nut_B_all))
add("  Timepoints with breast milk >80%, n",       lambda g: med1(g.nut_B_n))
add("Formula >80% at any timepoint, n (%)",        lambda g: pct(g.nut_F_any))
add("Nil by mouth at any timepoint, n (%)",        lambda g: pct(g.nut_NPO_any))
add("Birth-weight z-score at term-equivalent age", lambda g: med(g.bwz5))
add("  Change in z-score, T0 to T5",               lambda g: med(g.dz))
out=pd.DataFrame(rows_out, columns=["Characteristic","Overall (n = 43)","Unaffected (n = 38)","Affected (n = 5)"])
pd.set_option("display.width",180); pd.set_option("display.max_colwidth",46)
print(out.to_string(index=False))
print("\n--- 参考: T5検体が欠測の児 ---")
print(D.loc[D.t5_missing,["case","site","affected"]].to_string(index=False))
print("\n--- 抗菌薬パターン（発症児）---")
for _,r in D[D.affected].iterrows():
    print(f"  {r['case']:12s} {'/'.join(r[f'abx{t}'] for t in range(1,6))}   nut {'/'.join(r[f'nut{t}'] for t in range(1,6))}")
