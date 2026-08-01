import re, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
rcParams["font.family"]="Liberation Sans"; rcParams["pdf.fonttype"]=42; rcParams["axes.linewidth"]=0.8
rcParams["mathtext.fontset"]="custom"; rcParams["mathtext.rm"]="Liberation Sans"
rcParams["mathtext.it"]="Liberation Sans:italic"
GEN=["Bifidobacterium","Clostridium","Klebsiella","Enterococcus","Staphylococcus","Escherichia",
     "Lacticaseibacillus","Streptococcus"]
def ital(t):
    pat=r"\b(?:"+"|".join(sorted(GEN,key=len,reverse=True))+r")\b(?![a-z])"
    out=[];pos=0
    for m in re.finditer(pat,t):
        if m.start()>pos: out.append(t[pos:m.start()])
        out.append("$\\it{"+m.group(0)+"}$"); pos=m.end()
    if pos<len(t): out.append(t[pos:])
    return "".join(out)

cst=pd.read_pickle("cst_v2.pkl"); meta=pd.read_pickle("meta.pkl"); g=pd.read_pickle("genus.pkl")
NAME={7:"C1",1:"C2",4:"C3",2:"C4",0:"C5",5:"C6",6:"C7",3:"C8"}
COL={"C1":"#2E86AB","C2":"#E8613C","C3":"#F0A830","C4":"#2E6B4F","C5":"#8B72BE",
     "C6":"#D6336C","C7":"#3FB8AF","C8":"#8C6D46","C0":"#E8E8E8"}
DESC={"C1":"Bifidobacterium/Clostridium","C2":"Enterobacteriaceae/Klebsiella","C3":"Bifidobacterium",
      "C4":"Bifidobacterium/Enterococcus","C5":"Bifidobacterium/Staphylococcus","C6":"Escherichia",
      "C7":"Lacticaseibacillus","C8":"Staphylococcus","C0":"no analysable data"}
ORD=[f"C{i}" for i in range(1,9)]+["C0"]

d=meta[meta.siid.isin(set(g.siid))].merge(cst,on="siid"); d["P"]=d.cluster.map(NAME)
inf=meta.groupby("case").agg(site=("site","first"), ga=("ga","first"), ae=("ae_dol","first")).reset_index()
# 匿名ID: 施設別・在胎週数順
inf=inf.sort_values(["site","ga","case"]).reset_index(drop=True)
inf["ord"]=(~inf.ae.isna()).astype(int)
inf=inf.sort_values(["site","ord","ga"],ascending=[True,False,True]).reset_index(drop=True)
inf["aid"]=inf.groupby("site").cumcount().add(1).astype(str).str.zfill(2).radd(inf.site)

CW,CH=1.0,1.0
fig,ax=plt.subplots(figsize=(8.6,12.4))
ymap={}
y=0; sepy=[]
for site in ["A","B"]:
    sub=inf[inf.site==site]
    for _,r in sub.iterrows():
        ymap[r["case"]]=y; y+=1
    sepy.append(y); y+=1.1
tot=y-1.1

for _,r in inf.iterrows():
    c=r["case"]; yy=ymap[c]
    for t in range(6):
        row=d[(d.case==c)&(d.t==t)]
        st="C0" if len(row)==0 else row.P.iloc[0]
        post = (not np.isnan(r["ae"])) and len(row)>0 and row.dol.iloc[0]>=r["ae"]
        ax.add_patch(patches.Rectangle((t*CW,tot-yy-1),CW*0.92,CH*0.86,facecolor=COL[st],
                     edgecolor="black" if post else "white", lw=1.5 if post else 0.6))
        if st!="C0":
            ax.text(t*CW+CW*0.46,tot-yy-1+CH*0.43,st[1],ha="center",va="center",
                    fontsize=7.4,color="white",fontweight="bold")
    lab=r["aid"]+(f"  onset DOL {int(r['ae'])}" if not np.isnan(r["ae"]) else "")
    ax.text(-0.18,tot-yy-1+CH*0.43,lab,ha="right",va="center",fontsize=7.6,
            fontweight="bold" if not np.isnan(r["ae"]) else "normal")
ax.set_xlim(-2.6,6*CW+0.2); ax.set_ylim(-0.4,tot+0.6)
ax.set_xticks([t*CW+CW*0.46 for t in range(6)]); ax.set_xticklabels([f"T{t}" for t in range(6)],fontsize=9.5)
ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
ax.set_yticks([]); ax.tick_params(axis="x",length=0)
for sp in ("top","right","left","bottom"): ax.spines[sp].set_visible(False)
nA=(inf.site=="A").sum()
ax.text(-2.55,tot-nA/2+0.2,"Site A",rotation=90,va="center",ha="center",fontsize=10,fontweight="bold")
ax.text(-2.55,(tot-nA-1.1)/2-0.3,"Site B",rotation=90,va="center",ha="center",fontsize=10,fontweight="bold")
h=[patches.Patch(facecolor=COL[k],edgecolor="white",label=ital(f"{k} ({DESC[k]})")) for k in ORD]
h.append(patches.Patch(facecolor="white",edgecolor="black",lw=1.5,label="Sample at or after onset"))
ax.legend(handles=h,fontsize=7.8,frameon=False,loc="upper left",bbox_to_anchor=(0.0,-0.012),ncol=2)
fig.savefig("SupplementaryFigureS1_revised.pdf",bbox_inches="tight")
fig.savefig("figS1_prev.png",dpi=110,bbox_inches="tight")
print("児数:", len(inf), " Site A:", nA, " Site B:", (inf.site=="B").sum(), " 発症:", int(inf.ae.notna().sum()))
print("匿名ID例:", inf.aid.tolist()[:6], "...", inf.aid.tolist()[-3:])
