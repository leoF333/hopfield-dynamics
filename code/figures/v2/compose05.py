
# --- repository path bootstrap (added when this tree was packaged for release) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
for _p in (_R, _R.parent / "core", _R.parent / "experiments"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --------------------------------------------------------------------------------

import sys; sys.path.insert(0,'/tmp/p2')
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pathlib import Path
import glob
F="/sessions/clever-amazing-keller/mnt/figure_panel_rebuild"
OUTW=glob.glob('/sessions/*/mnt/dossier*')[0]+"/agent redacteur/figures_v2/"
def pil_font(size,bold=False):
    for c in (["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold
              else ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]):
        if Path(c).exists(): return ImageFont.truetype(c,size=size)
    return ImageFont.load_default()
def add_letter(im,letter):
    r=im.copy(); d=ImageDraw.Draw(r)
    f=pil_font(max(28,round(min(r.size)*0.050)),bold=True)
    t="(%s)"%letter; bb=d.textbbox((0,0),t,font=f)
    w,h=bb[2]-bb[0],bb[3]-bb[1]; p=max(8,round(h*0.25))
    d.rounded_rectangle((4,4,4+w+2*p,4+h+2*p),radius=p,fill="white",outline="#D4DAE0",width=max(1,p//5))
    d.text((4+p,4+p-bb[1]),t,font=f,fill="#151F28")
    return r
def compose(filename,title,rows,row_heights,images):
    width=5400; margin=70; gutter=44; header=155
    tf=pil_font(64,bold=True)
    H=header+margin+sum(row_heights)+gutter*(len(rows)-1)
    cv=Image.new("RGB",(width,H),"white"); d=ImageDraw.Draw(cv)
    d.text((margin,45),title,font=tf,fill="#17212B")
    d.line((margin,128,width-margin,128),fill="#CBD3DB",width=3)
    li=0; y=header
    for row,rh in zip(rows,row_heights):
        n=len(row); cw=(width-2*margin-gutter*(n-1))//n
        for c,key in enumerate(row):
            x=margin+c*(cw+gutter)
            im=add_letter(images[key],chr(ord("a")+li)); li+=1
            ct=ImageOps.contain(im,(cw,rh),Image.Resampling.LANCZOS)
            cv.paste(ct,(x+(cw-ct.width)//2,y+(rh-ct.height)//2))
            d.rectangle((x,y,x+cw,y+rh),outline="#D7DEE5",width=2)
        y+=rh+gutter
    for outdir in (OUTW, F+"/panels/"):
        try: cv.save(outdir+filename,dpi=(300,300),optimize=True)
        except Exception as e: print("  !",outdir,e)
    print("saved",filename,cv.size)

need=["n2a_without_seed51_raw","depinning_left","n2_c","depinning_right","pacemaker_full"]
images={k:Image.open(F+"/components/%s.png"%k).convert("RGB") for k in need}
images["floquet"]=ImageOps.expand(
    Image.open(OUTW+"component_floquet_hyperstability.png").convert("RGB"),
    border=(0,105,0,0),fill="white")
compose("Panel05_snic_characterization_v2.png",
        "The collective transition is a saddle-node on an invariant circle",
        [["n2a_without_seed51_raw","floquet","depinning_left"],
         ["n2_c","depinning_right"],
         ["pacemaker_full"]],
        [1120,1120,900], images)
