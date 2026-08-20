
# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import sys; sys.path.insert(0,'/tmp/p2')
from common2 import *
import matplotlib.gridspec as gsp, csv, json

L1=np.load(NC+'4_attracteurs_bassins_chaos/data/E23A_lyap.npz',allow_pickle=True)
GEO=np.load(NC+'4_attracteurs_bassins_chaos/data/E23A_geometry.npz',allow_pickle=True)
AG=np.load(NC+'4_attracteurs_bassins_chaos/data/E33_aging_mlx.npz',allow_pickle=True)
AG8=np.load(NC+'4_attracteurs_bassins_chaos/data/E33_aging_mlx_a0p08.npz',allow_pickle=True)
AG10=np.load(NC+'4_attracteurs_bassins_chaos/data/E33_aging_mlx_a0p10.npz',allow_pickle=True)
B1=np.load(NC+'4_attracteurs_bassins_chaos/data/E20a_basin_fractions.npz',allow_pickle=True)
B2=np.load(NC+'4_attracteurs_bassins_chaos/data/E20a_basin_fractions_hi.npz',allow_pickle=True)
R=list(csv.DictReader(open(PV+'/reports/N6_multiseed_lyapunov.csv')))
SP={}
for r in R: SP.setdefault((int(r['seed']),float(r['dt'])),{})[int(r['index'])]=float(r['lyapunov_exponent'])
seeds=sorted({k[0] for k in SP}); dts=sorted({k[1] for k in SP})
LMIN,LSTAR=0.2195,0.32763

fig=plt.figure(figsize=(19.0,13.6))
G=gsp.GridSpec(3,3,hspace=0.40,wspace=0.31,left=0.055,right=0.975,bottom=0.055,top=0.930)
suptitle(fig,"The window between the two transitions is filled by a genuine hyperchaotic attractor")

# --------------------------------------------------- (a) multi-seed spectra
ax=fig.add_subplot(G[0,0]); cm=dict(zip(seeds,plt.cm.viridis(np.linspace(0.05,0.85,len(seeds)))))
for (sd,dt),v in sorted(SP.items()):
    x=np.array(sorted(v)); y=np.array([v[i] for i in x]); fill=abs(dt-max(dts))<1e-12
    ax.plot(x,y,("o-" if fill else "^--"),color=cm[sd],ms=7 if fill else 6,lw=1.9,
            mfc=cm[sd] if fill else "none",alpha=1.0 if fill else 0.8,
            label=("seed %d"%sd) if fill else None,zorder=3)
ax.axhspan(0,0.075,color=GRN,alpha=0.07,zorder=0); ax.axhline(0,color="0.35",lw=1.1)
ax.set_xlabel(r"index $i$"); ax.set_ylabel(r"Lyapunov exponent $\Lambda_i$")
ax.set_xticks(range(1,9)); ax.set_ylim(-0.052,0.072)
ax.legend(frameon=False,ncol=2,columnspacing=0.8,loc="upper right",handlelength=1.3,fontsize=11.5)
ax.text(2.6,-0.036,"three exponents positive\nin every realization",fontsize=12.5,color=GRN,ha="center")
ax.text(0.975,0.035,r"filled $dt=0.01$ · open $dt=0.005$",transform=ax.transAxes,ha="right",
        fontsize=11.5,color=GREY)
L(ax,r"(a)  five realizations at $\lambda=0.31$")

# --------------------------------------------------- (b) spectrum + D_KY
ax=fig.add_subplot(G[0,1])
A2=np.asarray(L1['A2_lyaps'],float); cum=np.asarray(L1['A2_cum'],float); DKY=float(L1['A2_DKY'])
x=np.arange(1,len(A2)+1)
ax.bar(x,A2,color=[GRN if v>0 else "0.72" for v in A2],edgecolor="0.3",lw=0.8,zorder=2)
ax.axhline(0,color="0.35",lw=1.1); ax.set_xticks(x)
ax.set_xlabel(r"index $i$"); ax.set_ylabel(r"$\Lambda_i$",color=GRN); ax.tick_params(axis="y",colors=GRN)
b=ax.twinx(); b.plot(x,cum,"s--",color=INK,ms=6,lw=1.9,zorder=4); b.axhline(0,color=INK,lw=0.9,ls=":")
b.set_ylabel("cumulative sum",color=INK); b.tick_params(axis="y",colors=INK)
b.annotate(r"$D_{KY}\simeq%.2f$"%DKY,xy=(DKY,0),xytext=(0.30,0.30),textcoords="axes fraction",
           fontsize=13.5,color=INK,ha="center",arrowprops=dict(arrowstyle="->",color=INK,lw=1.4))
ax.text(0.03,0.06,"eight exponents do not close the sum:\nan order of magnitude, not a dimension",
        transform=ax.transAxes,ha="left",va="bottom",fontsize=11.5,color=GREY)
L(ax,"(b)  spectrum and cumulative sum")

# --------------------------------------------------- (c) lambda_L across the window
ax=fig.add_subplot(G[0,2])
lams=np.asarray(L1['A1_lams'],float); lys=np.asarray(L1['A1_lyaps'],float)
ax.axvspan(LMIN,LSTAR,color="0.90",zorder=0)
ax.axhspan(0,0.075,color=GRN,alpha=0.07,zorder=0)
ax.plot(lams,lys,"o-",ms=8,lw=2.1,color=INK,zorder=3)
ax.axhline(0,color="0.35",lw=1.1)
ax.axvline(LSTAR,color=RED,lw=1.4,ls="--",zorder=2)
ax.set_xlabel(r"mixing $\lambda$"); ax.set_ylabel(r"$\Lambda_1$")
ax.set_ylim(-0.004,0.072); ax.set_xlim(0.283,0.331)
ax.text(LSTAR-0.002,0.066,r"$\lambda^{*}$",color=RED,fontsize=14,ha="right",va="top")
ax.text(0.5,0.955,"positive across the whole window",transform=ax.transAxes,ha="center",va="top",
        fontsize=12.5,color=GRN)
L(ax,"(c)  not a property of one parameter value")

# --------------------------------------------------- (d) finite-time convergence
ax=fig.add_subplot(G[1,0])
ft=np.asarray(L1['A1b_ft'],float); lam_inf=float(L1['A1b_lyap'])
ax.plot(ft[:,0],ft[:,1],lw=1.6,color=INK)
ax.axhline(lam_inf,color=RED,lw=1.6,ls="--")
ax.axhline(0,color="0.5",lw=1.0,ls=":")
ax.set_xlabel("integration time  $T$"); ax.set_ylabel(r"finite-time $\Lambda_1(T)$")
ax.set_xlim(0,ft[:,0].max()); ax.set_ylim(min(-0.06,ft[:,1].min()*1.05),max(0.09,ft[:,1].max()*1.1))
ax.text(0.97,0.12,r"converges to $+%.4f$"%lam_inf+"\nover %d Lyapunov times"%int(ft[:,0].max()*lam_inf),
        transform=ax.transAxes,ha="right",va="bottom",fontsize=12.5,color=RED)
ax.text(0.03,0.94,"the positive exponent is not a transient:\nthe trajectory never comes to rest",
        transform=ax.transAxes,va="top",fontsize=12,color="#4c5a63")
L(ax,"(d)  it is an attractor, not a long transient")

# --------------------------------------------------- (e) return map
ax=fig.add_subplot(G[1,1])
pk=np.asarray(L1['A3_peaks'],float)
ax.plot(pk[:-1],pk[1:],"o",ms=8,color=INK,mfc="none",mew=1.6,alpha=0.85)
lo,hi=pk.min()-0.06,pk.max()+0.06
ax.plot([lo,hi],[lo,hi],"--",color="0.55",lw=1.2)
ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_aspect("equal")
ax.set_xlabel(r"$X_n$   (successive maxima of PC 1)"); ax.set_ylabel(r"$X_{n+1}$")
ax.text(0.03,0.95,"a dispersed cloud, not a one-dimensional curve:\n"
                  "no low-dimensional map reproduces it",
        transform=ax.transAxes,va="top",fontsize=12,color="#4c5a63")
L(ax,"(e)  the return map is not a curve")

# --------------------------------------------------- (f) PCA attractor
ax=fig.add_subplot(G[1,2])
x_,y_=np.asarray(GEO['x'],float),np.asarray(GEO['y'],float)
var=np.asarray(L1['A3_var'],float)
ax.plot(x_,y_,lw=0.35,color=INK,alpha=0.55)
ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
ax.text(0.03,0.95,"variance %.2f / %.2f / %.2f\non the first three components"%(var[0],var[1],var[2]),
        transform=ax.transAxes,va="top",fontsize=12,color="#4c5a63")
L(ax,"(f)  the attractor, projected")

# --------------------------------------------------- (g) no ageing
ax=fig.add_subplot(G[2,0])
tws=[0,50,200,800,3200]; cmw=plt.cm.plasma(np.linspace(0.08,0.78,len(tws)))
for c,tw in zip(cmw,tws):
    lag=np.asarray(AG['lag_%d'%tw],float); C=np.asarray(AG['C_%d'%tw],float)
    m=lag<=60
    ax.plot(lag[m],C[m]/C[m][0],lw=2.0,color=c,label=r"$t_w=%d$"%tw)
ax.set_xlabel(r"lag  $t-t_w$"); ax.set_ylabel(r"$C(t,t_w)\,/\,C(t_w,t_w)$")
ax.legend(frameon=False,ncol=2,columnspacing=0.9,loc="upper right",fontsize=11.5,handlelength=1.3)
ax.set_xlim(0,60)
ax.text(0.03,0.035,"the curves superpose in the lag and fail to collapse\n"
                  r"under $t/t_w$ — a stationary SRB measure, no ageing",
        transform=ax.transAxes,va="bottom",fontsize=12,color="#4c5a63")
L(ax,"(g)  two-time correlations do not age")

# --------------------------------------------------- (h) no ageing vs load
ax=fig.add_subplot(G[2,1])
def spread20(Z):
    v=[]
    for tw in (50,200,800,3200):                       # t_w = 0 excluded: still in the settle transient
        lag=np.asarray(Z['lag_%d'%tw],float); C=np.asarray(Z['C_%d'%tw],float)
        v.append(np.interp(20.0,lag,C/C[0]))
    return max(v)-min(v)
al=[0.05,0.08,0.10]; sp=[spread20(AG),spread20(AG8),spread20(AG10)]
nk=[int(AG['nkeep_tot']),int(AG8['nkeep_tot']),int(AG10['nkeep_tot'])]
ax.bar(range(3),sp,width=0.55,color=INK,edgecolor="none",zorder=3)
ax.axhline(0.1,color=RED,lw=1.9,ls="--",zorder=4)
ax.text(2.45,0.104,"ageing threshold",color=RED,fontsize=12.5,ha="right",va="bottom")
ax.set_xticks(range(3)); ax.set_xticklabels([r"$\alpha=%.2f$"%a for a in al])
ax.set_ylabel(r"spread of $C$ over $t_w$  (at lag $20$)")
ax.set_ylim(0,0.135)
for i,(v,n) in enumerate(zip(sp,nk)):
    ax.text(i,v+0.004,"%.3f\n$n=%d$"%(v,n),ha="center",va="bottom",fontsize=12,color=DARK)
ax.text(0.5,0.60,"every value is far below the threshold:\nno ageing emerges on the way to capacity\n"
                 r"(at $\alpha=0.12$ the chaotic set vanishes at $\lambda=0.31$)",
        transform=ax.transAxes,ha="center",va="center",fontsize=12,color="#4c5a63")
L(ax,"(h)  and none appears at higher load")

# --------------------------------------------------- (i) basins
ax=fig.add_subplot(G[2,2])
c1=json.loads(str(B1['counts'])); c2=json.loads(str(B2['counts']))
allc={**c1,**c2}
lam=np.array(sorted(float(k) for k in allc))
def frac(key):
    out=[]
    for l in lam:
        d=allc["%.4f"%l]; tot=sum(d.values())
        out.append((d.get('memory',0)+d.get('front',0))/tot if key=='static'
                   else d.get(key,0)/tot)
    return np.array(out)
st,cy_,ch=frac('static'),frac('cycle'),frac('chaos')
ot=1-st-cy_-ch
ax.stackplot(lam,st,ch,cy_,np.clip(ot,0,None),
             colors=[INK,"#b0483c",GRN,"0.85"],
             labels=["static memory fixed points","chaotic sea","recall cycle","unclassified"],
             edgecolor="w",lw=0.6)
ax.axvline(LMIN,color=RED,lw=1.6,ls="--"); ax.axvline(LSTAR,color=INK,lw=1.6,ls="--")
ax.set_xlim(lam.min(),lam.max()); ax.set_ylim(0,1)
ax.set_xlabel(r"mixing $\lambda$"); ax.set_ylabel("fraction of initial conditions")
ax.legend(frameon=True,framealpha=0.93,edgecolor="none",loc="lower left",fontsize=11.5,
          bbox_to_anchor=(0.005,0.13))
ax.text(LMIN-0.010,0.885,r"$\min_\mu\lambda_c$",color=RED,fontsize=13,ha="right",va="top")
ax.text(LSTAR+0.008,0.955,r"$\lambda^{*}$",color="w",fontsize=13.5,ha="left",va="top")
L(ax,"(i)  who owns phase space across the window")
save(fig,"Panel01_lyapunov_chaos_v2")
