import re
import pandas as pd, numpy as np, matplotlib, pickle
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.family"]="Liberation Sans"; rcParams["pdf.fonttype"]=42; rcParams["axes.linewidth"]=0.8
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


X=pd.read_pickle("feature_table.pkl"); g=pd.read_pickle("genus.pkl").set_index("siid")
den=g[[c for c in g.columns if c not in ("Sample","total_all")]].sum(axis=1)
rel=X.div(den,axis=0)*100; rel=rel.where(rel>=0.01,0.0)
keep=[c for c in rel.columns if rel[c].mean()>=0.01 and (rel[c]>0).mean()>=0.10]
P=rel[keep]; P=P.div(P.sum(axis=1),axis=0)
cst=pd.read_pickle("cst_v2.pkl").set_index("siid")["cluster"].reindex(P.index)
NAME={7:"C1",1:"C2",4:"C3",2:"C4",0:"C5",5:"C6",6:"C7",3:"C8"}
lab=cst.map(NAME).values
D,ks,avg,msil,store=pickle.load(open("sil_cache.pkl","rb"))
med=store[8][0]

# --- PCoA ---
n=len(D); J=np.eye(n)-np.ones((n,n))/n; B=-0.5*J@(D**2)@J
w,v=np.linalg.eigh(B); idx=np.argsort(w)[::-1]; w=w[idx]; v=v[:,idx]
pos=w[w>0]; pct=pos/pos.sum()*100
co=v[:,:2]*np.sqrt(np.maximum(w[:2],0))

COL={"C1":"#2E86AB","C2":"#E8613C","C3":"#F0A830","C4":"#2E6B4F",
     "C5":"#8B72BE","C6":"#D6336C","C7":"#3FB8AF","C8":"#8C6D46"}
DESC={"C1":"Bifidobacterium/Clostridium","C2":"Enterobacteriaceae/Klebsiella","C3":"Bifidobacterium",
      "C4":"Bifidobacterium/Enterococcus","C5":"Bifidobacterium/Staphylococcus","C6":"Escherichia",
      "C7":"Lacticaseibacillus","C8":"Staphylococcus"}
order=[f"C{i}" for i in range(1,9)]

fig=plt.figure(figsize=(12.6,4.6))
gs=fig.add_gridspec(1,2,width_ratios=[1.05,1],wspace=0.24)
ax=fig.add_subplot(gs[0,0])
for t in order:
    m=lab==t
    ax.scatter(co[m,0],co[m,1],s=17,color=COL[t],lw=0,alpha=0.9)
ax.scatter(co[med,0],co[med,1],s=62,facecolors="none",edgecolors="black",lw=1.3,zorder=5)
ax.axhline(0,color="#bbbbbb",lw=0.6); ax.axvline(0,color="#bbbbbb",lw=0.6)
ax.set_xlabel(f"PCo1 ({pct[0]:.1f}%)",fontsize=10); ax.set_ylabel(f"PCo2 ({pct[1]:.1f}%)",fontsize=10)
ax.tick_params(labelsize=9)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.text(-0.13,1.03,"a",transform=ax.transAxes,fontsize=14,fontweight="bold")

# --- 積み上げ棒 ---
ax2=fig.add_subplot(gs[0,1])
GEN=["Bifidobacterium","Klebsiella","Escherichia","Staphylococcus","Enterococcus",
     "Lacticaseibacillus","Streptococcus","Clostridium","Corynebacterium","Unclassified Enterobacteriaceae"]
GCOL=["#2E7D32","#E53935","#EF6C00","#9C6FB6","#2C4F9E","#4DB6AC","#F48FB1","#8D6E63","#BDBDBD","#C62828"]
prof=(P*100).assign(t=lab).groupby("t").mean().reindex(order)
mat=pd.DataFrame({gname:(prof[gname] if gname in prof else 0) for gname in GEN})
mat["Other"]=100-mat.sum(axis=1)
bottom=np.zeros(len(order))
for gname,c in list(zip(GEN,GCOL))+[("Other","#E0E0E0")]:
    ax2.bar(range(len(order)),mat[gname].values,bottom=bottom,width=0.72,color=c,lw=0,
            label=(ital(gname) if gname not in ("Unclassified Enterobacteriaceae","Other") else ("Unclassified\nEnterobacteriaceae" if gname.startswith("Unclass") else "Other")))
    bottom+=mat[gname].values
cnt=pd.Series(lab).value_counts().reindex(order)
ax2.set_xticks(range(len(order)))
ax2.set_xticklabels([f"{t}\n(n = {cnt[t]})" for t in order],fontsize=8.6)
ax2.set_ylim(0,100); ax2.set_ylabel("Mean relative abundance (%)",fontsize=10); ax2.tick_params(labelsize=9)
for sp in ("top","right"): ax2.spines[sp].set_visible(False)
ax2.legend(fontsize=6.7,frameon=False,loc="upper left",bbox_to_anchor=(1.01,1.02),
           title="Genus",title_fontsize=7.4,labelspacing=0.55)
ax2.text(-0.16,1.03,"b",transform=ax2.transAxes,fontsize=14,fontweight="bold")

# 型の凡例（パネルa下）
handles=[plt.Line2D([],[],marker="o",ls="",ms=5.5,color=COL[t],label=ital(f"{t} ({DESC[t]})")) for t in order]
handles.append(plt.Line2D([],[],marker="o",ls="",ms=7,mfc="none",mec="black",label="Medoid"))
fig.legend(handles=handles,loc="lower left",bbox_to_anchor=(0.055,-0.16),ncol=3,frameon=False,fontsize=7.6)
fig.savefig("Figure1_revised.pdf",bbox_inches="tight")
fig.savefig("fig1_prev.png",dpi=140,bbox_inches="tight")
print("PCo1 %.1f%%  PCo2 %.1f%%" % (pct[0],pct[1]))
print("クラスタサイズ:", cnt.to_dict())
