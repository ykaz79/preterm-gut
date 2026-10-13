# Analysis code

Analysis code for *"Longitudinal Transitions of Gut Microbial Community Types in Preterm Infants Receiving Probiotics: An Exploratory Two-Site Cohort Study."*

All analyses reported in the manuscript were performed with these scripts.

---

## Environment

Python 3.12.3

| Library | Version | Used for |
|---|---|---|
| numpy | 2.4.4 | numerical computation |
| pandas | 3.0.2 | data handling |
| scipy | 1.17.1 | Fisher's exact test, chi-squared test, Mann–Whitney U |
| statsmodels | 0.14.6 | linear mixed models, generalised estimating equations |
| scikit-learn | 1.8.0 | area under the ROC curve |
| matplotlib | 3.10.8 | figures |
| openpyxl | 3.1.5 | reading the clinical data workbook |

Jensen–Shannon distance, partitioning around medoids (BUILD + SWAP), silhouette
statistics, principal coordinate analysis and the exact patient-level permutation
test were implemented directly in `numpy` rather than taken from a library, so
that every step is inspectable. No R code was used for any reported result.

---

## Input files

| File | Content |
|---|---|
| `bar_chart_3_genus.csv` | genus-level read counts, 206 samples x 326 taxa (delivered by the sequencing laboratory) |
| `bar_chart_4_family.csv` | family-level read counts, 206 samples x 105 taxa |
| `BGM研究ノート学会短縮版_v2.xlsx` | clinical data, 43 infants (transposed: rows = variables, columns = infants) |
| `SIID48410read.xlsx` | per-sample read counts at each processing stage |

Scripts expect these under `/mnt/user-data/uploads/`; edit the paths at the top of
`build_data.py` and `nd_decompose.py` if the files are stored elsewhere.

---

## Order of execution

Scripts write intermediate objects as pickles in the working directory. Run in
this order; later scripts depend on the pickles written by earlier ones.

### 1. Data construction

| Script | Output | Purpose |
|---|---|---|
| `build_data.py` | `meta.pkl`, `genus.pkl`, `family.pkl` | parses the clinical workbook into long format (infant x timepoint), assigns sequencing batch from sample ID, reconciles clinical records with the count tables |
| `nd_decompose.py` | `feature_table.pkl` | derives the genus-to-family mapping from the two count tables and reassigns reads labelled "Not determined" to family level, producing the feature table used throughout |

### 2. Community states

| Script | Output | Purpose |
|---|---|---|
| `cst2.py` | `cst_v2.pkl` | abundance and prevalence filtering (21 features retained), Jensen–Shannon distance, PAM over k = 2–12, silhouette statistics, eight-cluster assignment |
| `grid_cst.py` | — | preprocessing sensitivity grid (denominator, order of thresholding, prevalence rule, distance definition) |
| `pcoa_test.py` | — | principal coordinate analysis; compares the metric properties of the Jensen–Shannon distance and the divergence |
| `k_sens.py` | — | Supplementary Table S1: membership of the Enterobacteriaceae-dominant type across k = 2–12 |

### 3. Abundance comparisons and inference

| Script | Output | Purpose |
|---|---|---|
| `reproduce.py` | `d.pkl` | relative abundance, pre-onset group definition, Table 2 effect sizes |
| `batch_models.py` | `d_final.pkl` | batch diagnostics, linear mixed models and generalised estimating equations with batch as a five-level variable, batch-stratified comparisons |
| `perm.py` | — | exact patient-level permutation tests (unstratified, site-restricted, batch-stratified) |
| `perm_all.py` | — | permutation P values for all six taxa in Table 2 |
| `major6.py` | — | consolidated Site B analyses reported in the Results |

### 4. Specificity, transitions and site effects

| Script | Output | Purpose |
|---|---|---|
| `c2_table.py` | — | infant-level 2 x 2 tables, sensitivity, specificity, predictive values |
| `retention.py` | — | self-retention of each community type, stratified by outcome |
| `c2_unaffected.py` | `supp_c2.pkl` | Supplementary Table S2: clinical characteristics of unaffected infants assigned to C2 |
| `site_pgct.py` | — | Supplementary Table S3: site x community type |
| `xtab.py` | — | Supplementary Table S4: batch composition and its association with site |
| `siteB_recluster.py` | — | full workflow repeated within Site B |
| `circularity.py`, `auc_table.py` | `auc_tab.pkl` | Supplementary Table S5: discriminative performance, with cluster bootstrap confidence intervals |
| `table1_ext.py` | — | additional Table 1 variables derived from the clinical workbook |

### 5. Figures

| Script | Output |
|---|---|
| `fig1.py` | Figure 1 (principal coordinate analysis; mean composition per community type) |
| `fig23.py` | Figure 2 (alluvial diagram) and Figure 3 (trajectories of affected infants) |
| `fig4.py` | Figure 4 (pre-onset versus unaffected abundance) |
| `figS1.py` | Supplementary Figure S1 (individual trajectories, all 43 infants) |
| `supp_fig_sil.py` | Supplementary Figure S2 (silhouette statistics across k) |

`supp_fig_sil.py` writes `sil_cache.pkl`, which caches the distance matrix and all
PAM solutions; `fig1.py` and `k_sens.py` read this cache, so run it before them.

---

## Notes on implementation

**Distance.** The Jensen–Shannon distance is the square root of the Jensen–Shannon
divergence, computed with base-2 logarithms. The square root is used because the
divergence itself does not satisfy the triangle inequality; `pcoa_test.py`
quantifies this (5.29 per cent of randomly sampled triples violate the inequality
for the divergence, none for the distance) and shows that principal coordinate
analysis on the divergence yields negative eigenvalues accounting for 22.95 per
cent of the total absolute eigenvalue mass.

**Permutation test.** Permutation is performed over infants, not samples: all
longitudinal observations from an infant move together. All allocations are
enumerated exhaustively (962,598 for the whole cohort, 42,504 within Site B,
59,840 for the batch-stratified scheme, 5,280 for Site B stratified by batch).
The test statistic is the difference in mean relative abundance between the
samples of infants allocated to the complication group and those of all remaining
infants.

**Bootstrap.** Confidence intervals for areas under the ROC curve are obtained by
resampling infants with replacement (4,000 replicates), so that the within-infant
correlation is preserved.

**Reproducibility.** Random number use is limited to the bootstrap in
`auc_table.py` (seed 20260726), the Monte Carlo contingency-table tests in
`xtab.py` (seed 0) and `pcoa_test.py` (seed 0), and point jitter in `fig4.py`
(seed 7). All other results are deterministic.
