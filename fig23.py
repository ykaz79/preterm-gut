import re
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
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


cst=pd.read_pickle("cst_v2.pkl"); meta=pd.read_pickle("meta.pkl"); g=pd.read_pickle("genus.pkl")
NAME={7:"C1",1:"C2",4:"C3",2:"C4",0:"C5",5:"C6",6:"C7",3:"C8"}
COL={"C1":"#2E86AB","C2":"#E8613C","C3":"#F0A830","C4":"#2E6B4F","C5":"#8B72BE",
     "C6":"#D6336C","C7":"#3FB8AF","C8":"#8C6D46","C0":"#D9D9D9"}
DESC={"C1":"Bifidobacterium/Clostridium","C2":"Enterobacteriaceae/Klebsiella","C3":"Bifidobacterium",
      "C4":"Bifidobacterium/Enterococcus","C5":"Bifidobacterium/Staphylococcus","C6":"Escherichia",
      "C7":"Lacticaseibacillus","C8":"Staphylococcus","C0":"no analysable data"}
ORD=[f"C{i}" for i in range(1,9)]+["C0"]

d=meta[meta.siid.isin(set(g.siid))].merge(cst,on="siid"); d["P"]=d.cluster.map(NAME)
cases=sorted(meta.case.unique())
state={}
for c in cases:
    for t in range(6):
        row=d[(d.case==c)&(d.t==t)]
        state[(c,t)]="C0" if len(row)==0 else row.P.iloc[0]

# ================= Figure 2: alluvial =================
fig,ax=plt.subplots(figsize=(10.2,5.4))
BW=0.30; GAP=1.2
xpos={t:t*(BW+GAP) for t in range(6)}
ycoord={}
for t in range(6):
    cnt=pd.Series([state[(c,t)] for c in cases]).value_counts()
    y=0.0; ycoord[t]={}
    for k in ORD:
        if k not in cnt: continue
        h=cnt[k]
        ax.add_patch(patches.Rectangle((xpos[t],y),BW,h,facecolor=COL[k],lw=0))
        ycoord[t][k]=[y,y+h]; y+=h+0.55
    tot=y
for t in range(5):
    flow={}
    for c in cases:
        a,b=state[(c,t)],state[(c,t+1)]
        flow[(a,b)]=flow.get((a,b),0)+1
    off_a={k:ycoord[t][k][0] for k in ycoord[t]}
    off_b={k:ycoord[t+1][k][0] for k in ycoord[t+1]}
    for (a,b) in sorted(flow, key=lambda x:(ORD.index(x[0]),ORD.index(x[1]))):
        n=flow[(a,b)]
        y0,y1=off_a[a],off_a[a]+n; off_a[a]+=n
        y2,y3=off_b[b],off_b[b]+n; off_b[b]+=n
        x0=xpos[t]+BW; x1=xpos[t+1]
        xm=(x0+x1)/2
        verts=[(x0,y0),(xm,y0),(xm,y2),(x1,y2),(x1,y3),(xm,y3),(xm,y1),(x0,y1),(x0,y0)]
        codes=[Path.MOVETO,Path.CURVE4,Path.CURVE4,Path.CURVE4,Path.LINETO,
               Path.CURVE4,Path.CURVE4,Path.CURVE4,Path.CLOSEPOLY]
        ax.add_patch(patches.PathPatch(Path(verts,codes),facecolor=COL[a],lw=0,alpha=0.42))
ax.set_xlim(-0.3,xpos[5]+BW+0.3); ax.set_ylim(-1.5,tot)
ax.set_xticks([xpos[t]+BW/2 for t in range(6)]); ax.set_xticklabels([f"T{t}" for t in range(6)],fontsize=10)
ax.set_yticks([]); ax.set_xlabel("Postnatal sampling period",fontsize=10.5)
for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
h=[patches.Patch(facecolor=COL[k],label=ital(f"{k} ({DESC[k]})")) for k in ORD]
ax.legend(handles=h,fontsize=7.8,frameon=False,loc="upper left",bbox_to_anchor=(1.005,1.0))
fig.savefig("Figure2_revised.pdf",bbox_inches="tight"); fig.savefig("fig2_prev.png",dpi=140,bbox_inches="tight")

# ================= Figure 3: 症例タイムライン =================
aff=d[d.ae_dol.notna()].copy()
info=aff.groupby("case").agg(ae=("ae_dol","first")).sort_values("ae")
# NEC 症例の識別：発症例のうち、最終検体が発症後で以降の追跡がない症例（原稿 Fig.3 の Case 3）
# 実データでは腸管合併症の種別が研究ノートに記録されている。ここでは種別列から読み込む。
NEC_CASES = set(aff.loc[aff.get("comp_type","")=="NEC","case"]) if "comp_type" in aff.columns else set()
if not NEC_CASES:
    # 種別列が無い場合は、T5 検体を欠く発症例を NEC とみなす（当該症例は発症直後に死亡）
    NEC_CASES = {c for c in info.index if 5 not in set(d.loc[d.case==c,"t"])}
TYPE = {c: "NEC" for c in NEC_CASES}
fig,ax=plt.subplots(figsize=(10.2,4.4))
rows=list(info.index)
for i,c in enumerate(rows):
    y=len(rows)-1-i
    sub=d[d.case==c].sort_values("t"); ae=info.loc[c,"ae"]
    xs=(sub.dol-ae).values
    ax.plot(xs,[y]*len(xs),"-",color="#999999",lw=0.9,zorder=1)
    for x,st,t in zip(xs,sub.P,sub.t):
        post=x>=0
        ax.scatter([x],[y],s=210,color=COL[st],edgecolors="black" if post else "none",
                   lw=1.3 if post else 0,zorder=3)
        ax.text(x,y,st[1],ha="center",va="center",fontsize=8,color="white",zorder=4,fontweight="bold")
        ax.text(x,y-0.34,f"T{int(t)}",ha="center",va="top",fontsize=6.6,color="#666666")
    if c in NEC_CASES:
        ax.annotate("died after onset",(xs[-1],y),xytext=(xs[-1]+7,y+0.02),fontsize=7.5,color="#B22222",va="center")
ax.axvline(0,color="black",ls="--",lw=1.1)
ax.text(0,len(rows)-0.35,"Onset",ha="center",fontsize=9,fontweight="bold")
ax.axvspan(-52,0,color="#EFF4F8",zorder=0)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f"{'Case '+str(i+1)} ({TYPE.get(c,'ileus')})\n(onset DOL {int(info.loc[c,'ae'])})"
                    for i,c in enumerate(rows)][::-1],fontsize=8.2)
ax.set_xlabel("Days relative to symptom onset",fontsize=10.5)
ax.set_xlim(-52,95); ax.set_ylim(-0.7,len(rows)-0.2)
for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
ax.tick_params(axis="y",length=0)
present=sorted({state[(c,t)] for c in rows for t in range(6) if state[(c,t)]!="C0"}, key=ORD.index)
h=[patches.Patch(facecolor=COL[k],label=ital(f"{k} ({DESC[k]})")) for k in present]
ax.legend(handles=h,fontsize=7.6,frameon=False,loc="upper left",bbox_to_anchor=(1.005,1.0),title="Community type",title_fontsize=8)
fig.savefig("Figure3_revised.pdf",bbox_inches="tight"); fig.savefig("fig3_prev.png",dpi=140,bbox_inches="tight")
print("Fig2/Fig3 保存")
print("\n発症例の型推移:")
for c in rows:
    sub=d[d.case==c].sort_values("t")
    print(f"  {c:12s} onset {int(info.loc[c,'ae']):3d}  " + " ".join(f"T{int(t)}:{p}" for t,p in zip(sub.t,sub.P)))
