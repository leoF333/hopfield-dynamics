
# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import numpy as np, glob, os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
 "font.size":13,"font.family":"DejaVu Sans","axes.linewidth":1.1,
 "xtick.direction":"in","ytick.direction":"in","xtick.top":True,"ytick.right":True,
 "xtick.major.size":4.5,"ytick.major.size":4.5,"xtick.major.width":1.0,"ytick.major.width":1.0,
 "xtick.labelsize":12,"ytick.labelsize":12,
 "axes.labelsize":14,"legend.fontsize":12,"axes.titlesize":15,
 "lines.linewidth":1.9,"lines.markersize":6.5,"lines.markeredgewidth":1.1,
 "legend.handlelength":1.6,"legend.borderpad":0.3,"legend.labelspacing":0.32,
 "mathtext.fontset":"dejavusans","figure.dpi":100,"savefig.dpi":300})
NC="/sessions/clever-amazing-keller/mnt/numerics copie/results/"
PV=glob.glob('/sessions/*/mnt/paper_v5_numerics')[0]
FPR="/sessions/clever-amazing-keller/mnt/figure_panel_rebuild"
OUTW=glob.glob('/sessions/*/mnt/dossier*')[0]+"/agent redacteur/figures_v2/"
os.makedirs(OUTW,exist_ok=True)
INK="#1f4e79"; RED="#c0392b"; GRN="#2e8b57"; ORA="#d1670a"; GREY="#7c8a94"; PUR="#6b4c9a"; DARK="#22313a"
def L(ax,t,fs=15,pad=7): ax.set_title(t,fontsize=fs,loc="left",pad=pad)
def tag(fig,ax,s,dx=-0.085,dy=1.045):
    ax.text(dx,dy,s,transform=ax.transAxes,fontsize=17,fontweight="bold",va="bottom",ha="left")
def suptitle(fig,s):
    fig.suptitle(s,fontsize=19,fontweight="bold",x=0.012,ha="left",y=0.995)
def save(fig,name):
    for d in (OUTW, FPR+"/panels/"):
        try:
            fig.savefig(d+name+".png",dpi=300,bbox_inches="tight",pad_inches=0.12,facecolor="white")
        except Exception as e: print("  ! could not write to",d,e)
    print("saved",name,fig.get_size_inches())
