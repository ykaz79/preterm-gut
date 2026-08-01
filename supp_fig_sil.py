import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.family"]="Liberation Sans"   # Arial 代替（metric互換）
rcParams["pdf.fonttype"]=42; rcParams["ps.fonttype"]=42
rcParams["axes.linewidth"]=0.8

X0=pd.read_pickle("feature_table.pkl"); g=pd.read_pickle("genus.pkl").set_index("siid")
den=g[[c for c in g.columns if c not in ("Sample","total_all")]].sum(axis=1)
rel=X0.div(den,axis=0)*100; rel=rel.where(rel>=0.01,0.0)
keep=[c for c in rel.columns if rel[c].mean()>=0.01 and (rel[c]>0).mean()>=0.10]
P=rel[keep]; P=P.div(P.sum(axis=1),axis=0)

def jsd(P):
    A=P.values;n=len(A);D=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            p,q=A[i],A[j];m=0.5*(p+q);a=p>0;b=q>0
            d=0.5*np.sum(p[a]*np.log2(p[a]/m[a]))+0.5*np.sum(q[b]*np.log2(q[b]/m[b]))
            D[i,j]=D[j,i]=np.sqrt(max(d,0.))
    return D
def pam(D,k):
    n=len(D);med=[int(np.argmin(D.sum(1)))]
    while len(med)<k:
        cur=D[:,med].min(1);best,bg=None,-np.inf
        for c in range(n):
            if c in med:continue
            gn=np.maximum(cur-D[:,c],0).sum()
            if gn>bg:bg,best=gn,c
        med.append(best)
    imp=True
    while imp:
        imp=False;cost=D[:,med].min(1).sum()
        for mi in range(k):
            for h in range(n):
                if h in med:continue
                tr=med.copy();tr[mi]=h;c=D[:,tr].min(1).sum()
                if c<cost-1e-12:med,cost,imp=tr,c,True
    return np.array(med),np.argmin(D[:,med],1)
def sil_vals(D,lab):
    n=len(D);s=np.zeros(n)
    for i in range(n):
        same=(lab==lab[i]).copy();same[i]=False
        if same.sum()==0: continue
        a=D[i,same].mean();b=min(D[i,lab==c].mean() for c in set(lab) if c!=lab[i])
        s[i]=(b-a)/max(a,b)
    return s
def medoid_sil(D,med):
    d=np.sort(D[:,med],axis=1)
    return np.nanmean(1-d[:,0]/np.where(d[:,1]==0,np.nan,d[:,1]))

import os, pickle
if os.path.exists("sil_cache.pkl"):
    D,ks,avg,msil,store=pickle.load(open("sil_cache.pkl","rb")); _cached=True
else:
    D=jsd(P); _cached=False
if not _cached:
    ks=list(range(2,13)); avg=[]; msil=[]; store={}
    for k in ks:
        med,lab=pam(D,k); sv=sil_vals(D,lab)
        avg.append(sv.mean()); msil.append(medoid_sil(D,med)); store[k]=(med,lab,sv)
    pickle.dump((D,ks,avg,msil,store), open("sil_cache.pkl","wb"))
print(pd.DataFrame({"k":ks,"avg":np.round(avg,4),"medoid":np.round(msil,4)}).to_string(index=False))

NAME={0:"C5",1:"C2",2:"C4",3:"C8",4:"C3",5:"C6",6:"C7",7:"C1"}
COL={"C1":"#2E86AB","C2":"#E8613C","C3":"#F0A830","C4":"#2E6B4F",
     "C5":"#8B72BE","C6":"#D6336C","C7":"#3FB8AF","C8":"#8C6D46"}

fig=plt.figure(figsize=(10.5,4.2))
gs=fig.add_gridspec(1,2,width_ratios=[1,1.12],wspace=0.28)

ax=fig.add_subplot(gs[0,0])
ax.plot(ks[1:],avg[1:],"-o",color="#333333",ms=5,lw=1.4,label="Average silhouette width")
ax.plot(ks[1:],msil[1:],"-s",color="#8899AA",ms=4.5,lw=1.2,label="Medoid silhouette")
ax.plot([2],[avg[0]],"o",mfc="white",mec="#333333",ms=5,lw=1.4)
ax.plot([2],[msil[0]],"s",mfc="white",mec="#8899AA",ms=4.5)
ax.plot([2,3],[avg[0],avg[1]],":",color="#333333",lw=1.0)
ax.plot([2,3],[msil[0],msil[1]],":",color="#8899AA",lw=1.0)
i8=ks.index(8)
ax.plot(8,avg[i8],"o",ms=11,mfc="none",mec="#E8613C",mew=1.6)
ax.annotate("k = 8",(8,avg[i8]),xytext=(8.5,avg[i8]+0.055),fontsize=9.5,color="#E8613C",
            arrowprops=dict(arrowstyle="-",color="#E8613C",lw=0.9))
ax.set_xticks(ks); ax.set_xlabel("Number of clusters (k)"); ax.set_ylabel("Silhouette statistic")
ax.set_ylim(0.03,0.47); ax.set_xlim(1.4,12.6)
ax.legend(frameon=False,fontsize=8.5,loc="upper right")
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.text(-0.13,1.04,"a",transform=ax.transAxes,fontsize=13,fontweight="bold")

ax2=fig.add_subplot(gs[0,1])
med,lab,sv=store[8]
order=sorted(range(8),key=lambda c:-int((lab==c).sum()))
y=0; yt=[]; ylab=[]
for c in order:
    v=np.sort(sv[lab==c])
    ax2.barh(np.arange(y,y+len(v)),v,height=1.0,color=COL[NAME[c]],lw=0)
    yt.append(y+len(v)/2); ylab.append(f"{NAME[c]} (n={len(v)})")
    y+=len(v)+6
ax2.axvline(0,color="#999999",lw=0.7)
ax2.set_yticks(yt); ax2.set_yticklabels(ylab,fontsize=9)
ax2.set_xlabel("Silhouette width"); ax2.set_ylim(-4,y); ax2.invert_yaxis()
ax2.set_xlim(-0.25,0.85)
for sp in ("top","right","left"): ax2.spines[sp].set_visible(False)
ax2.tick_params(axis="y",length=0)
ax2.text(-0.20,1.04,"b",transform=ax2.transAxes,fontsize=13,fontweight="bold")

fig.savefig("SupplementaryFigure_silhouette.pdf",bbox_inches="tight")
fig.savefig("supp_sil.png",dpi=150,bbox_inches="tight")
print("saved")
