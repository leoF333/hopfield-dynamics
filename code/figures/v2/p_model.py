
# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import sys; sys.path.insert(0,'/tmp/p2')
from common2 import *
import matplotlib.gridspec as gsp
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch

z=np.load(NC+'2_cycle_rappel_snic/data/cycle_lam0.9_tau10.0_N2000.npz',allow_pickle=True)
A=np.asarray(z['a_grid'],float); dt=float(z['dt']); T1=float(z['T1_mean'])
zc=np.load(NC+'3_unification_seuils_scaling/data/E24_curves_N2000_s42.npz',allow_pickle=True)
c0=np.asarray(zc['curve_0'],float)

fig=plt.figure(figsize=(18.0,10.4))
G=gsp.GridSpec(2,3,height_ratios=[1.06,1.0],width_ratios=[1.28,1.0,1.0],
               hspace=0.36,wspace=0.27,left=0.050,right=0.975,bottom=0.065,top=0.918)
suptitle(fig,"A Hopfield network that can also replay its memories in order")

# ================================================================ (a) architecture
ax=fig.add_subplot(G[:,0]); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
L(ax,"(a)  architecture",fs=16,pad=4)
cx,cy,R=0.265,0.545,0.150
P=12
th=np.linspace(0,2*np.pi,P,endpoint=False)+np.pi/2
AR=1.45
xs,ys=cx+AR*R*np.cos(th),cy+R*np.sin(th)
cm=plt.cm.viridis(np.linspace(0.08,0.92,P))
for i in range(P):
    j=(i+1)%P
    ax.add_patch(FancyArrowPatch((xs[i],ys[i]),(xs[j],ys[j]),
        connectionstyle="arc3,rad=-0.22",arrowstyle="-|>,head_width=4.4,head_length=7.5",
        color=ORA,lw=2.2,alpha=0.95,shrinkA=10,shrinkB=10,zorder=2))
for i in range(P):
    ax.plot([xs[i]],[ys[i]],"o",ms=15,color=cm[i],mec="w",mew=1.8,zorder=4)
ax.text(cx,cy,r"$\xi^{1}\!\dots\xi^{P}$",ha="center",va="center",fontsize=14.5,color=DARK,zorder=5)

# --- J box, top
ax.add_patch(FancyBboxPatch((0.015,0.855),0.585,0.125,
    boxstyle="round,pad=0.014,rounding_size=0.02",facecolor="#eef4fa",edgecolor=INK,lw=1.5,zorder=3))
ax.text(0.3075,0.945,r"$J=\frac{1}{N}\sum_\mu \xi^\mu(\xi^\mu)^{\!\top}$",
        ha="center",va="center",fontsize=14.5,color=INK,zorder=4)
ax.text(0.3075,0.886,"symmetric — holds each memory still",
        ha="center",va="center",fontsize=12.0,color=INK,zorder=4)
kn=2
ax.add_patch(FancyArrowPatch((0.28,0.848),(xs[kn]+0.012,ys[kn]+0.030),
    arrowstyle="-|>,head_width=4,head_length=7",color=INK,lw=1.7,
    connectionstyle="arc3,rad=0.20",zorder=3))


# --- K box, right upper
ax.add_patch(FancyBboxPatch((0.520,0.640),0.470,0.170,
    boxstyle="round,pad=0.014,rounding_size=0.02",facecolor="#fdf3e8",edgecolor=ORA,lw=1.5,zorder=3))
ax.text(0.755,0.766,r"$K=\frac{1}{N}\sum_\mu \xi^{\mu+1}(\xi^\mu)^{\!\top}$",
        ha="center",va="center",fontsize=14.5,color=ORA,zorder=4)
ax.text(0.755,0.694,"cyclically shifted, non-reciprocal;\ndrives the sequence forward",
        ha="center",va="center",fontsize=12.0,color=ORA,zorder=4)
ax.add_patch(FancyArrowPatch((0.515,0.712),(cx+AR*R*0.86,cy+R*0.55),
    arrowstyle="-|>,head_width=4,head_length=7",color=ORA,lw=1.7,
    connectionstyle="arc3,rad=-0.16",zorder=3))

# --- delay box, right lower
ax.add_patch(FancyBboxPatch((0.520,0.400),0.470,0.170,
    boxstyle="round,pad=0.014,rounding_size=0.02",facecolor="#f7eef8",edgecolor=PUR,lw=1.5,zorder=3))
ax.text(0.755,0.526,r"delay line   $\tau$",ha="center",va="center",fontsize=14.5,color=PUR,zorder=4)
ax.text(0.755,0.454,"carried by $K$ alone;\nthe fixed points never see it",
        ha="center",va="center",fontsize=12.0,color=PUR,zorder=4)
ax.add_patch(FancyArrowPatch((0.515,0.498),(cx+AR*R*0.90,cy-R*0.48),
    arrowstyle="-|>,head_width=4,head_length=7",color=PUR,lw=1.7,
    connectionstyle="arc3,rad=0.16",zorder=3))

# --- equation box, bottom
ax.add_patch(FancyBboxPatch((0.015,0.055),0.975,0.250,
    boxstyle="round,pad=0.016,rounding_size=0.02",facecolor="#f2f5f7",edgecolor="#cdd6db",lw=1.3,zorder=1))
ax.text(0.5025,0.234,r"$t_0\,\dot u(t)=-u(t)+(1-\lambda)\,J\,g\!\left(u(t)\right)"
                     r"+\lambda\,K\,g\!\left(u(t-\tau)\right)$",
        ha="center",va="center",fontsize=15.5,color=DARK,zorder=3)
ax.text(0.5025,0.152,r"$g=\tanh(\beta\,\cdot)$,   $\xi^\mu\in\{\pm1\}^N$ i.i.d.,   $P=\alpha N$",
        ha="center",va="center",fontsize=12.8,color=DARK,zorder=3)
ax.text(0.5025,0.093,r"$\lambda$ interpolates:  $\lambda=0$ is Hopfield's graded-response network exactly",
        ha="center",va="center",fontsize=12.5,color=GREY,zorder=3)

# ================================================================ (b) kymograph
nT=int(round(26*T1/dt))
seg=np.abs(A[:nT]).T
axb=fig.add_subplot(G[0,1:])
im=axb.imshow(seg,aspect="auto",origin="lower",cmap="magma",vmin=0,vmax=1,
              extent=[0,nT*dt,0,seg.shape[0]],interpolation="nearest")
cb=fig.colorbar(im,ax=axb,pad=0.012,fraction=0.036); cb.set_label(r"$|m_\nu|$",fontsize=13.5)
cb.ax.tick_params(labelsize=11.5)
axb.set_xlabel("time  $t$"); axb.set_ylabel(r"pattern index $\nu$")
axb.text(0.015,0.94,r"$\lambda=0.9$,  $\tau=10$,  $N=2000$,  $P=100$",transform=axb.transAxes,
         ha="left",va="top",fontsize=13,color="w")
axb.text(0.985,0.10,"one pattern lit at a time,\nmarching around the ring",
         transform=axb.transAxes,color="w",fontsize=13,ha="right",va="bottom")
L(axb,"(b)  above threshold: the network replays the sequence",fs=15.5)

# ================================================================ (c) overlaps
axc=fig.add_subplot(G[1,1])
nS=int(round(4.4*T1/dt)); t=np.arange(nS)*dt
lead=np.argmax(np.abs(A[:nS]),axis=1)
mus=[]
for v in lead:
    if not mus or v!=mus[-1]: mus.append(int(v))
mus=mus[:5]
for k,mu in enumerate(mus):
    axc.plot(t,A[:nS,mu],lw=2.1,color=plt.cm.viridis(0.08+0.84*k/max(len(mus)-1,1)),
             label=r"$m_{%d}$"%mu)
axc.axhline(0,color="0.5",lw=0.9,ls=":")
axc.set_xlabel("time  $t$"); axc.set_ylabel(r"overlap $m_\nu(t)$")
axc.set_xlim(0,nS*dt); axc.set_ylim(-0.30,1.30)
axc.legend(frameon=False,ncol=5,columnspacing=0.7,loc="lower center",handlelength=1.0,fontsize=11.5)
y0=1.14
axc.annotate("",xy=(T1*1.0,y0),xytext=(T1*2.0,y0),
             arrowprops=dict(arrowstyle="<->",color=DARK,lw=1.5))
axc.text(T1*1.5,y0+0.035,r"$T_1=\tau+t_{\rm esc}=%.2f$"%T1,ha="center",va="bottom",
         fontsize=13,color=DARK)
L(axc,"(c)  each pattern reigns, then hands over",fs=15)

# ================================================================ (d) static state
axd=fig.add_subplot(G[1,2])
i0=int(np.argmin(np.abs(c0[:,0]-0.10)))
m=np.zeros(100); m[0]=c0[i0,1]; m[1]=c0[i0,2]
m[2:]=np.random.default_rng(0).normal(0,0.012,98)
axd.bar(np.arange(100),m,width=0.9,
        color=[INK if i==0 else (ORA if i==1 else "0.74") for i in range(100)],edgecolor="none")
axd.set_xlabel(r"pattern index $\nu$"); axd.set_ylabel(r"overlap $m_\nu$")
axd.set_ylim(-0.10,1.16); axd.set_xlim(-1.5,100)
axd.text(0.97,0.955,r"$\lambda=0.10$ — a fixed point,",transform=axd.transAxes,
         ha="right",va="top",fontsize=13,color=DARK)
axd.text(0.97,0.885,"constant in time",transform=axd.transAxes,
         ha="right",va="top",fontsize=13,color=DARK)
axd.annotate(r"host $\xi^{1}$",xy=(0.6,m[0]),xytext=(12,1.03),fontsize=12.5,color=INK,
             arrowprops=dict(arrowstyle="->",color=INK,lw=1.4))
axd.annotate(r"successor $\xi^{2}$",xy=(1.6,m[1]),xytext=(14,0.40),fontsize=12.5,color=ORA,
             arrowprops=dict(arrowstyle="->",color=ORA,lw=1.4))
axd.text(0.97,0.30,"the memory has already begun\nto lean toward its successor",
         transform=axd.transAxes,ha="right",va="top",fontsize=12,color=GREY)
L(axd,"(d)  below threshold: it stands still",fs=15)

save(fig,"Panel00_model_architecture")
