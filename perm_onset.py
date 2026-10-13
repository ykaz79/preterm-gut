"""Exact patient-level permutation test with onset-age censoring (Supplementary Methods).

Under the null, every allocation of k infants to the complication group is combined
with every assignment of the k observed ages at onset (k! orderings).  For an
allocated infant only samples obtained before its assigned onset age enter the
complication group; all samples of the remaining infants form the comparison group.
The statistic is the difference in mean relative abundance (complication minus
comparison).  The observed statistic is computed with the same rule, so it equals
the pre-onset versus without-complications comparison reported in Table 2.

Requires meta.pkl, genus.pkl and family.pkl written by build_data.py.
"""
import itertools
import numpy as np
import pandas as pd

meta = pd.read_pickle("meta.pkl")
g = pd.read_pickle("genus.pkl").set_index("siid")
f = pd.read_pickle("family.pkl").set_index("siid")

# Relative abundance (%), denominator = all genus-level reads (as in batch_models.py)
gcols = [c for c in g.columns if c not in ("Sample", "total_all")]
den = g[gcols].sum(axis=1)
GEN = ["Klebsiella", "Escherichia", "Enterobacter", "Serratia"]
rel = pd.DataFrame(index=g.index)
for c in GEN:
    rel[c] = g[c] / den * 100
rel["Summed_Entero"] = f["Enterobacteriaceae"] / den * 100
rel["unclassified_Entero"] = rel["Summed_Entero"] - rel[GEN].sum(axis=1)

# All 206 analysed samples (samples at or after onset are needed under the null)
D = meta.merge(rel.reset_index(), on="siid", how="inner").sort_values(["case", "t"])
D["batch"] = D["batch"].astype(int)
ONSET = D.groupby("case")["ae_dol"].first()          # NaN for infants without complications


def exact_perm(data, var, strat=None, comparison_filter=None):
    """Return (observed difference, number of allocations, two-sided P).

    comparison_filter: optional boolean Series (aligned with data) selecting the samples
    that may enter the comparison group (used for the time-matched analyses).
    """
    cases = sorted(data.case.unique())
    ix = {c: i for i, c in enumerate(cases)}
    aff = [c for c in cases if not np.isnan(ONSET[c])]
    ages = np.array(sorted(ONSET[aff].astype(float)))
    k = len(aff)
    keep = (np.ones(len(data), bool) if comparison_filter is None
            else comparison_filter.loc[data.index].values)
    x = data[var].values; dol = data.dol.values; ci = data.case.map(ix).values
    n_c = len(cases)
    # per-infant totals for the comparison group
    s_all = np.bincount(ci, weights=x * keep, minlength=n_c)
    n_all = np.bincount(ci, weights=keep.astype(float), minlength=n_c)
    # per-infant sums of samples obtained before each candidate onset age
    s_pre = np.zeros((n_c, k)); n_pre = np.zeros((n_c, k))
    for j, a in enumerate(ages):
        m = dol < a
        s_pre[:, j] = np.bincount(ci, weights=x * m, minlength=n_c)
        n_pre[:, j] = np.bincount(ci, weights=m.astype(float), minlength=n_c)
    S, N = s_all.sum(), n_all.sum()

    # observed
    obs_i = np.array([ix[c] for c in aff])
    obs_age = np.array([np.searchsorted(ages, ONSET[c]) for c in aff])
    a1 = s_pre[obs_i, obs_age].sum(); m1 = n_pre[obs_i, obs_age].sum()
    obs = a1 / m1 - (S - s_all[obs_i].sum()) / (N - n_all[obs_i].sum())

    # allocations of infants
    if strat is None:
        combs = np.array(list(itertools.combinations(range(n_c), k)), dtype=np.int32)
    else:
        layer = data.groupby("case")[strat].first()
        need = layer[aff].value_counts().to_dict()
        parts = [list(itertools.combinations([ix[c] for c in cases if layer[c] == b], kk))
                 for b, kk in sorted(need.items())]
        combs = np.array([list(itertools.chain.from_iterable(p))
                          for p in itertools.product(*parts)], dtype=np.int32)
    A_all = s_all[combs].sum(1); M_all = n_all[combs].sum(1)
    ge = 0; total = 0
    for perm in itertools.permutations(range(k)):          # k! assignments of onset ages
        a1 = sum(s_pre[combs[:, j], perm[j]] for j in range(k))
        m1 = sum(n_pre[combs[:, j], perm[j]] for j in range(k))
        with np.errstate(invalid="ignore", divide="ignore"):
            st = a1 / m1 - (S - A_all) / (N - M_all)
        ge += np.sum(np.abs(st) >= abs(obs) - 1e-9)
        total += len(combs)
    return obs, total, ge / total


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    TAXA = [("Summed Enterobacteriaceae", "Summed_Entero"), ("Klebsiella", "Klebsiella"),
            ("Escherichia", "Escherichia"), ("unclassified Enterobacteriaceae", "unclassified_Entero"),
            ("Enterobacter", "Enterobacter"), ("Serratia", "Serratia")]
    B = D[D.site == "B"]
    rows = []
    def add(label, taxon, var, data, **kw):
        obs, n, p = exact_perm(data, var, **kw)
        rows.append(dict(analysis=label, taxon=taxon, diff_pp=round(obs, 2), allocations=f"{n:,}", P=round(p, 4)))

    for t, v in TAXA:                                       # Table 2
        add("Whole cohort", t, v, D)
    for t, v in TAXA[:2]:                                   # Supplementary Results
        add("Site B", t, v, B)
        add("Whole cohort, batch-stratified", t, v, D, strat="batch")
        add("Site B, batch-stratified", t, v, B, strat="batch")
    # Supplementary Table S7: comparison group restricted in time
    le46 = (D.dol <= 46)
    for t, v in TAXA[:2]:
        add("Term-equivalent samples excluded", t, v, D, comparison_filter=(D.t != 5))
        add("Day of life <= 46", t, v, D, comparison_filter=le46)
        add("Site B, day of life <= 46", t, v, B, comparison_filter=le46)
    # Sensitivity analysis: infant with the shortest interruption of feeding (onset day 10) removed
    case1 = ONSET.dropna().idxmin()
    for t, v in TAXA[:2]:
        add("Whole cohort, earliest-onset infant removed", t, v, D[D.case != case1])
    print(pd.DataFrame(rows).to_string(index=False))
