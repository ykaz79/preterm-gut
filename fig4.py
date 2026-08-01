import re
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.family"]="Liberation Sans"; rcParams["pdf.fonttype"]=42; rcParams["ps.fonttype"]=42
rcParams["axes.linewidth"]=0.8
rcParams["mathtext.fontset"]="custom"
rcParams["mathtext.rm"]="Liberation Sans"
rcParams["mathtext.it"]="Liberation Sans:italic"
rcParams["mathtext.bf"]="Liberation Sans:bold"
_GEN=['Bifidobacterium', 'Clostridium', 'Klebsiella', 'Enterococcus', 'Staphylococcus', 'Escherichia', 'Lacticaseibacillus', 'Streptococcus', 'Corynebacterium', 'Lactobacillus', 'Lactococcus', 'Veillonella', 'Rothia', 'Finegoldia', 'Cutibacterium', 'Blautia', 'Limosilactobacillus', 'Collinsella', 'Peptacetobacter', 'Enterobacter', 'Serratia']
def ital(t):
    """属名のみ斜体にした mathtext 文字列を返す（科名・株名は正体）"""
    pat=r"\b(?:"+"|".join(sorted(_GEN,key=len,reverse=True))+r")\b(?![a-z])"
    out=[];pos=0
    for m in re.finditer(pat,t):
        if m.start()>pos: out.append(t[pos:m.start()])
        out.append("$\\it{"+m.group(0)+"}$"); pos=m.end()
    if pos<len(t): out.append(t[pos:])
    return "".join(out)


d=pd.read_pickle("d_final.pkl")
pre=d[d.y==1]; un=d[d.y==0]
print("pre-onset", len(pre), " unaffected", len(un))

def cliffs(a,b):
    a=np.asarray(a); b=np.asarray(b)
    return ((a[:,None]>b[None,:]).sum()-(a[:,None]<b[None,:]).sum())/(len(a)*len(b))
def cohend(a,b):
    a=np.asarray(a); b=np.asarray(b)
    s=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    return (a.mean()-b.mean())/s

PANELS=[("a","Summed Enterobacteriaceae","Summed_Entero",False,
         "P = 0.0009 (LMM); 0.0002 (GEE); 0.0014 (permutation)"),
        ("b","Klebsiella","Klebsiella",True,
         "P = 0.0003 (permutation)")]
PINK="#F2B8B5"; PINKPT="#D9534F"; GREY="#D9D9D9"; GREYPT="#9E9E9E"
fig,axes=plt.subplots(1,2,figsize=(7.4,4.4)); rng=np.random.default_rng(7)

for ax,(lab,title,col,ital,ptxt) in zip(axes,PANELS):
    a=pre[col].values; b=un[col].values
    bp=ax.boxplot([a,b],positions=[1,2],widths=0.55,patch_artist=True,showfliers=False,
                  medianprops=dict(color="black",lw=1.4),
                  whiskerprops=dict(color="black",lw=0.8),capprops=dict(color="black",lw=0.8),
                  boxprops=dict(lw=0.8,edgecolor="black"))
    bp["boxes"][0].set_facecolor(PINK); bp["boxes"][1].set_facecolor(GREY)
    for x0,v,c in [(1,a,PINKPT),(2,b,GREYPT)]:
        ax.scatter(x0+rng.uniform(-0.17,0.17,len(v)),v,s=11,color=c,alpha=0.75,lw=0,zorder=3)
    ax.set_xticks([1,2]); ax.set_xticklabels([f"Pre-onset\n(n = {len(a)})",f"Unaffected\n(n = {len(b)})"],fontsize=9)
    ax.set_ylim(-4,100); ax.set_yticks(range(0,101,20))
    ax.set_ylabel("Relative abundance (%)",fontsize=9.5)
    ax.tick_params(labelsize=9)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.set_title(title,fontsize=10.5,style="italic" if ital else "normal",pad=26)
    ax.text(0.5,1.055,f"Cliff's \u03b4 = {cliffs(a,b):.2f}; Cohen's d = {cohend(a,b):.1f}",
            transform=ax.transAxes,ha="center",fontsize=8.2)
    ax.text(0.5,1.005,ptxt,transform=ax.transAxes,ha="center",fontsize=7.6)
    ax.text(-0.20,1.10,lab,transform=ax.transAxes,fontsize=13,fontweight="bold")
    print(f"panel {lab}: delta={cliffs(a,b):.2f} d={cohend(a,b):.1f}")

plt.tight_layout(w_pad=2.4)
fig.savefig("Figure4_revised.pdf",bbox_inches="tight")
fig.savefig("fig4_prev.png",dpi=150,bbox_inches="tight")
print("saved")
