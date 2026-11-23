import argparse, csv, json, platform
from pathlib import Path
import numpy as np
import networkx as nx
import numba
from numba import njit
from scipy.stats import t as student
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ===================== shared constants =====================
BASE=np.array([[[8.5,4.5],[1.,5.5]],[[9.,2.],[.5,1.]]]) # O/C, A/I, P/M
WIDTH=np.array([[[1.5,1.],[.6,1.]],[[1.2,.6],[.2,.3]]])
def matrix(flip_p=False,flip_m=False):
    a=BASE.copy()
    if flip_p: a[:,:,0]=BASE[::-1,:,0]
    if flip_m: a[:,:,1]=BASE[:,::-1,1]
    return a
CASES={'E1':BASE,'E2':matrix(True,True),'E3':matrix(False,True),'E4':matrix(True,False)}
TOPO={'BA–WS':('BA','WS'),'BA–BA':('BA','BA'),'WS–WS':('WS','WS')}
LABELS={'E1':'E1: substitutes','E2':'E2: complements','E3':'E3: P− / M+','E4':'E4: P+ / M−'}
COLORS=['#355C7D','#C17B49','#378D85','#8878A3']; STYLES=['-','--','-.',':']; MARKERS=['o','s','^','D']
METRICS=['platform_openness','sme_adoption','business_collaboration','matching_gap','platform_payoff','sme_payoff']
NP,NM=50,200

# ===================== shared model =====================
def pack(g,rng):
    weights={tuple(sorted(e)):rng.uniform(.5,1.5) for e in sorted(g.edges())}
    n=len(g); deg=np.array([g.degree(i) for i in range(n)],dtype=np.int64)
    adj=np.zeros((n,max(deg)),dtype=np.int64); cum=np.zeros_like(adj,dtype=float)
    for i in range(n):
        ne=sorted(g.neighbors(i)); adj[i,:len(ne)]=ne
        if ne:
            w=np.array([weights[tuple(sorted((i,j)))] for j in ne]); cum[i,:len(ne)]=np.cumsum(w)/w.sum(); cum[i,len(ne)-1]=1.
    return adj,deg,cum

def setup(seed,topology):
    streams=np.random.SeedSequence(seed).spawn(5)
    rng=np.random.default_rng(streams[0]); business=rng.random((NP,NM))<.1; digital=rng.random((NP,NM))<.1
    for a in (business,digital):
        for i in np.flatnonzero(~a.any(axis=1)): a[i,rng.integers(NM)]=True
        for j in np.flatnonzero(~a.any(axis=0)): a[rng.integers(NP),j]=True
    weights=rng.random((NP,NM))*digital
    bi,bj=np.nonzero(business); dw_p=weights.sum(1)/digital.sum(1); dw_m=weights.sum(0)/digital.sum(0)
    graphs=[]
    for n,kind,m,k,s in zip((NP,NM),TOPO[topology],(3,2),(6,4),streams[1:3]):
        r=np.random.default_rng(s); gseed=int(r.integers(2**31-1)); g=nx.barabasi_albert_graph(n,m,seed=gseed) if kind=='BA' else nx.watts_strogatz_graph(n,k,.1,seed=gseed)
        graphs.append(pack(g,r))
    r=np.random.default_rng(streams[3]); sp=np.zeros(NP,np.int64); sm=np.zeros(NM,np.int64)
    sp[r.choice(NP,NP//2,False)]=1; sm[r.choice(NM,NM//2,False)]=1
    return bi,bj,dw_p,dw_m,*graphs[0],*graphs[1],sp,sm,streams[4]

@njit(cache=True)
def measure(sp,sm,bi,bj):
    x=sp.mean(); y=sm.mean(); c=0.
    for l in range(len(bi)): c+=sp[bi[l]]*sm[bj[l]]
    c/=len(bi)
    return x,y,c,c-x*y

@njit(cache=True)
def run(bi,bj,dwp,dwm,pa,pd,pc,ma,md,mc,sp,sm,means,width,noise,draws,alpha,lam,mu,bp,bm,theta,dp,dm):
    steps=draws.shape[0]; npop=len(sp); nm=len(sm); out=np.zeros((steps+1,6)); out[0,:4]=measure(sp,sm,bi,bj); out[0,4:]=np.nan
    cp=np.zeros(npop); cm=np.zeros(nm)
    for l in range(len(bi)): cp[bi[l]]+=1; cm[bj[l]]+=1
    for t in range(steps):
        up=np.zeros(npop); um=np.zeros(nm)
        for l in range(len(bi)):
            i,j=bi[l],bj[l]; a,b=1-sp[i],1-sm[j]
            up[i]+=means[a,b,0]+width[a,b,0]*noise[t,l,0]; um[j]+=means[a,b,1]+width[a,b,1]*noise[t,l,1]
        up=up/cp+alpha*dwp*dp*sp; um=um/cm+alpha*dwm*dm*sm
        out[t+1,4]=up.mean(); out[t+1,5]=um.mean()
        for q in range(npop+nm):
            r=draws[t,q]; focal=int(r[0]*(npop+nm))
            if focal<npop:
                i=focal; d=pd[i]
                if d==0: continue
                z=int(r[2]*d)
                if r[1]<lam:
                    z=0
                    while z<d-1 and r[2]>pc[i,z]: z+=1
                j=pa[i,z]; prob=1/(1+np.exp(-max(-40.,min(40.,bp*(up[j]-up[i])))))
                if r[3]<prob: sp[i]=sp[j]
            else:
                i=focal-npop; d=md[i]
                if d==0: continue
                if r[1]<mu:
                    j=ma[i,int(r[2]*d)]; prob=1/(1+np.exp(-max(-40.,min(40.,bm*(um[j]-um[i])))))
                    if r[3]<prob: sm[i]=sm[j]
                else:
                    mx=-1e100
                    for z in range(d): mx=max(mx,um[ma[i,z]])
                    total=0.
                    for z in range(d): total+=np.exp(theta*(um[ma[i,z]]-mx))
                    acc=0.; j=ma[i,d-1]
                    for z in range(d):
                        acc+=np.exp(theta*(um[ma[i,z]]-mx))
                        if acc>=r[2]*total: j=ma[i,z]; break
                    sm[i]=sm[j]
        out[t+1,:4]=measure(sp,sm,bi,bj)
    return out

def ci(a):
    m=np.mean(a,axis=0); h=student.ppf(.975,len(a)-1)*np.std(a,axis=0,ddof=1)/np.sqrt(len(a)) if len(a)>1 else np.zeros_like(m)
    return m,h

def style():
    fonts={f.name for f in font_manager.fontManager.ttflist}; font='Times New Roman' if 'Times New Roman' in fonts else 'Liberation Serif' if 'Liberation Serif' in fonts else 'DejaVu Serif'
    plt.rcParams.update({'font.family':font,'font.size':11,'axes.labelsize':12,'axes.linewidth':.8,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':9.5,'pdf.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'stix','figure.facecolor':'white','axes.facecolor':'white'})
    return font

def save(fig,folder,name):
    fig.savefig(folder/f'{name}.png',dpi=600,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)

def plot(data,folder,name,keys,labels,metrics,steps):
    fig,axs=plt.subplots(1,2,figsize=(7.1,3.65)); fig.subplots_adjust(left=.105,right=.985,bottom=.25,top=.88,wspace=.35)
    for z,(ax,k) in enumerate(zip(axs,metrics)):
        for i,key in enumerate(keys):
            m,h=ci(data[key][:,:,k]); x=np.arange(steps+1)
            ax.plot(x,m,color=COLORS[i],ls=STYLES[i],lw=1.7,marker=MARKERS[i],markevery=max(1,steps//10),ms=3,markerfacecolor='white',markeredgewidth=.65,label=labels[i]); ax.fill_between(x,np.clip(m-h,0,1) if k<3 else m-h,np.clip(m+h,0,1) if k<3 else m+h,color=COLORS[i],alpha=.10,lw=0)
        ax.set(xlim=(0,steps),xlabel='Monte Carlo steps',ylabel=['Platform openness','SME adoption','Business-link collaboration','Matching gap'][k]); ax.grid(color='#D9DEE3',lw=.45,alpha=.7); ax.set_axisbelow(True)
        if k<3: ax.set_ylim(-.025,1.025); ax.set_yticks(np.arange(0,1.01,.2))
        else: ax.axhline(0,color='#777777',lw=.7)
        ax.text(.5,-.31,f'({"ab"[z]})',ha='center',va='top',transform=ax.transAxes,fontsize=12)
    h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,loc='upper center',ncol=2 if len(keys)==4 else 3,frameon=False,bbox_to_anchor=(.5,.995),columnspacing=1.1,handlelength=2.4)
    if len(keys)==4: fig.subplots_adjust(top=.80)
    save(fig,folder,name)

# ===================== network-experiment helpers =====================
def cross(seed):
    rng=np.random.default_rng(np.random.SeedSequence(seed).spawn(5)[0]); b=rng.random((NP,NM))<.1; d=rng.random((NP,NM))<.1
    for a in (b,d):
        for i in np.flatnonzero(~a.any(1)): a[i,rng.integers(NM)]=True
        for j in np.flatnonzero(~a.any(0)): a[rng.integers(NP),j]=True
    return b,d,rng.random((NP,NM))*d

def matched(b,d,w,target,seed):
    rng=np.random.default_rng(np.random.SeedSequence([seed,339])); inside=rng.permutation(np.flatnonzero(b)); outside=rng.permutation(np.flatnonzero(~b)); n=int(d.sum()); k=round(n*target)
    if k>len(inside) or n-k>len(outside): raise ValueError('infeasible overlap')
    a=np.zeros(NP*NM,bool); a[inside[:k]]=True; a[outside[:n-k]]=True; a=a.reshape(NP,NM)
    for axis in (0,1):
        for v in np.flatnonzero(~a.any(axis=1-axis)):
            for u in rng.permutation(NM if axis==0 else NP):
                i,j=(v,u) if axis==0 else (u,v)
                donors=np.argwhere(a & (b==b[i,j]) & (a.sum(1)[:,None]>1) & (a.sum(0)[None,:]>1))
                if len(donors):
                    p,q=donors[rng.integers(len(donors))]; a[p,q]=False; a[i,j]=True; break
            else: raise RuntimeError('coverage repair failed')
    assert a.any(1).all() and a.any(0).all() and a.sum()==n and (a&b).sum()==k
    weights=np.zeros_like(w); weights[a]=np.random.default_rng(np.random.SeedSequence([seed,340])).permutation(w[d]); assert np.isclose(weights.sum(),w.sum())
    return weights.sum(1)/a.sum(1),weights.sum(0)/a.sum(0),k/n,a.sum(1).std(),a.sum(0).std()

# ===================== experiment 1: topology + incentive =====================
def run_topology(a,root,figs,tables,font):
    keys=[('BA–WS','E1'),('BA–BA','E1'),('WS–WS','E1'),('BA–WS','E2'),('BA–WS','E3'),('BA–WS','E4')]
    data={k:[] for k in keys}; seeds=[a.seed+r*12347 for r in range(a.reps)]
    for r,seed in enumerate(seeds):
        cache={}
        for top,case in keys:
            if top not in cache: cache[top]=setup(seed,top)
            *net,sp,sm,stream=cache[top]; rng=np.random.default_rng(stream); noise=rng.uniform(-1,1,(a.steps,len(net[0]),2)); draws=rng.random((a.steps,NP+NM,4))
            data[top,case].append(run(*net,sp.copy(),sm.copy(),CASES[case],WIDTH,noise,draws,a.alpha,.5,.5,1.,1.,1.,a.delta_p,a.delta_m))
        if (r+1)%10==0 or r+1==a.reps: print(f'[topology] {r+1}/{a.reps} paired replications',flush=True)
    data={k:np.stack(v) for k,v in data.items()}; W=min(40,a.steps)
    with (tables/'trajectories_topology.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['topology','case','rep','seed','step',*METRICS]);
        for (top,c),arr in data.items():
            for r,sim in enumerate(arr):
                for t,v in enumerate(sim): w.writerow([top,c,r+1,seeds[r],t,*[f'{x:.8f}' for x in v]])
    with (tables/'summary_topology.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['topology','case','metric','reps','window','mean','sd','ci95_low','ci95_high']);
        for (top,c),arr in data.items():
            for k,m in enumerate(METRICS):
                v=arr[:,-W:,k].mean(1); x,h=ci(v); w.writerow([top,c,m,a.reps,W,x,v.std(ddof=1),x-h,x+h])
    with (tables/'paired_contrasts_topology.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['topology','case','reference','metric','mean_difference','ci95_low','ci95_high']); ref=data[keys[0]][:,-W:,:].mean(1)
        for top,c in keys[1:]:
            for k,m in enumerate(METRICS):
                v=data[top,c][:,-W:,k].mean(1)-ref[:,k]; x,h=ci(v); w.writerow([top,c,'BA–WS/E1',m,x,x-h,x+h])
    with (tables/'payoff_matrices_topology.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['case','strategy','P_mean','P_low','P_high','M_mean','M_low','M_high']);
        for c,m in CASES.items():
            for i,j,s in [(0,0,'OA'),(0,1,'OI'),(1,0,'CA'),(1,1,'CI')]:
                p,q=m[i,j]; u,v=WIDTH[i,j]; w.writerow([c,s,p,p-u,p+u,q,q-v,q+v])
    (root/'parameters_topology.json').write_text(json.dumps({'experiment':'topology','reps':a.reps,'steps':a.steps,'seeds':seeds,'window':W,'alpha':a.alpha,'delta_p':a.delta_p,'delta_m':a.delta_m,'lambda':.5,'mu':.5,'beta_p':1,'beta_m':1,'theta':1,'population':[NP,NM],'initial_ratios':[.5,.5],'BA_m':[3,2],'WS_k':[6,4],'WS_rewire':.1,'business_probability':.1,'digital_probability':.1,'digital_weights':'U(0,1)','platform_learning_weights':'symmetric U(0.5,1.5)','versions':{'python':platform.python_version(),'numpy':np.__version__,'networkx':nx.__version__,'numba':numba.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},'font':font,'notes':'Stylized scenarios. Net payoff may be negative. Digital payoff differences and behavioral parameters are reconstruction assumptions, not verified original code. Paired common random numbers; independent graph and initialization per replication.'},indent=2,ensure_ascii=False),encoding='utf-8')
    plot(data,figs,'Fig01_network_evolution',keys[:3],[k[0] for k in keys[:3]],(0,1),a.steps)
    k2=[keys[0],*keys[3:]]; labs=[LABELS[k[1]] for k in k2]
    plot(data,figs,'Fig02_incentive_evolution',k2,labs,(0,1),a.steps)
    plot(data,figs,'Fig03_coordination',k2,labs,(2,3),a.steps)
    return seeds, W

# ===================== experiment 2: strength + overlap =====================
def run_network(a,root,figs,tables,font):
    alphas=[0,.25,.5,.75,1,1.5,2]; overlaps=[.1,.25,.4,.55,.7]; keys=[('strength',g,v) for g in [0,.7] for v in alphas]+[('overlap',.7,v) for v in overlaps]
    data={k:[] for k in keys}; seeds=[a.seed+r*12347 for r in range(a.reps)]; diagnostics=[]
    for r,seed in enumerate(seeds):
        *net,sp,sm,stream=setup(seed,'BA–WS'); b,d,w=cross(seed)
        assert np.array_equal(np.nonzero(b)[0],net[0]); assert np.allclose(w.sum(1)/d.sum(1),net[2])
        rng=np.random.default_rng(stream); noise=rng.uniform(-1,1,(a.steps,len(net[0]),2)); draws=rng.random((a.steps,NP+NM,4))
        for group,g,v in keys:
            m=BASE.copy(); m[0,0,:]+=g; n=list(net); alpha=v if group=='strength' else .5
            if group=='overlap':
                n[2],n[3],actual,sdP,sdM=matched(b,d,w,v,seed); diagnostics.append([r+1,seed,v,actual,int(d.sum()),float(w.sum()),sdP,sdM])
            data[group,g,v].append(run(*n,sp.copy(),sm.copy(),m,WIDTH,noise,draws,alpha,.5,.5,1.,1.,1.,1.,1.))
        if (r+1)%10==0 or r+1==a.reps: print(f'[network] {r+1}/{a.reps} replications',flush=True)
    data={k:np.stack(v) for k,v in data.items()}; W=40; hh=np.sqrt(np.log(40)/(2*a.reps))
    def write(name,header,rows):
        with (tables/name).open('w',newline='') as f:
            z=csv.writer(f); z.writerow(header); z.writerows(rows)
    write('network_diagnostics.csv',['rep','seed','target_overlap','actual_overlap','digital_edges','total_weight','degree_sd_P','degree_sd_M'],diagnostics)
    write('trajectories_network.csv',['group','g','parameter','rep','seed','step',*METRICS],([*k,r+1,seeds[r],t,*[f'{x:.8f}' for x in v]] for k,arr in data.items() for r,sim in enumerate(arr) for t,v in enumerate(sim)))
    rows=[]
    for k,arr in data.items():
        ref=data[(k[0],k[1],.5 if k[0]=='strength' else .1)][:,-W:,:].mean(1)
        for j,m in enumerate(METRICS):
            v=arr[:,-W:,j].mean(1); x,h=ci(v); delta,dh=ci(v-ref[:,j]); rows.append([*k,m,x,v.std(ddof=1),x-h,x+h,delta,delta-dh,delta+dh,max(0,x-hh) if j<3 else '',min(1,x+hh) if j<3 else ''])
    write('summary_network.csv',['group','g','parameter','metric','mean','sd','t_ci_low','t_ci_high','paired_difference','paired_ci_low','paired_ci_high','bounded_ci_low','bounded_ci_high'],rows)
    colors=['#355C7D','#378D85','#898D93','#C17B49','#8878A3']; styles=['-','--','-.',':','-']; markers=['o','s','^','D','v']
    for group,vals,selected,name,sym in [('strength',alphas,[0,.5,.75,1,2],'Fig07_empowerment','\\alpha'),('overlap',overlaps,overlaps,'Fig08_matching','\\omega')]:
        fig,axs=plt.subplots(1,2,figsize=(7.5,3.65)); fig.subplots_adjust(left=.105,right=.98,bottom=.25,top=.8,wspace=.36)
        for i,v in enumerate(selected):
            m,h=ci(data[group,.7,v][:,:,2]); t=np.arange(a.steps+1); axs[0].plot(t,m,color=colors[i],ls=styles[i],marker=markers[i],markevery=max(1,a.steps//10),ms=3,mfc='white',lw=1.7,label=f'${sym}={v:g}$'); axs[0].fill_between(t,np.clip(m-h,0,1),np.clip(m+h,0,1),color=colors[i],alpha=.09,lw=0)
        axs[0].set(xlim=(0,a.steps),xlabel='Monte Carlo steps',ylabel='Business-link collaboration'); axs[0].legend(loc='lower left',bbox_to_anchor=(0,1.03),ncol=3,frameon=False,fontsize=8,columnspacing=.7,handlelength=1.8)
        for i,g in enumerate([0,.7] if group=='strength' else [.7]):
            means=np.array([data[group,g,v][:,-W:,2].mean() for v in vals]); lo=np.maximum(0,means-hh); hi=np.minimum(1,means+hh); axs[1].errorbar(vals,means,yerr=[means-lo,hi-means],color=colors[i],ls=styles[i],marker=markers[i],mfc='white',lw=1.65,capsize=2.5,ms=4,label=f'$g={g:g}$')
        axs[1].set(xlabel='Digital empowerment intensity' if group=='strength' else 'Digital-edge overlap share',ylabel='Final-window collaboration'); axs[1].legend(frameon=False,loc='best',fontsize=9); axs[1].set_xticks(vals); axs[1].tick_params(axis='x',labelsize=9)
        for z,ax in enumerate(axs):
            ax.set_ylim(-.025,1.025); ax.set_yticks(np.arange(0,1.01,.2)); ax.grid(color='#D9DEE3',lw=.45,alpha=.7); ax.set_axisbelow(True); ax.text(.5,-.31,f'({"ab"[z]})',ha='center',transform=ax.transAxes,fontsize=12)
        save(fig,figs,name)
    (root/'parameters_network.json').write_text(json.dumps({'experiment':'network','reps':a.reps,'steps':a.steps,'seeds':seeds,'window':W,'cases':keys,'overlap_definition':'intersection / digital edge count','overlap_controls':'fixed business graph, digital edge count and weight multiset; no isolates; digital degree sequence varies','overlap_alpha':.5,'population':[NP,NM],'topology':'BA–WS','initial_ratios':[.5,.5],'lambda':.5,'mu':.5,'beta_P':1,'beta_M':1,'theta':1,'delta_P':1,'delta_M':1,'cost_changes':[0,0],'font':font,'notes':'g=.7 selected from Experiment 2 transition region. Digital payoff only depends on own strategy and local mean weight, not partner strategy. Trajectory t CI descriptive, endpoint Hoeffding pointwise.'},indent=2),encoding='utf-8')
    return seeds, W

# ===================== main =====================
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reps',type=int,default=100)
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--alpha',type=float,default=.5)
    p.add_argument('--delta-p',type=float,default=1.)
    p.add_argument('--delta-m',type=float,default=1.)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'experiment3_results')
    a=p.parse_args()
    if a.reps<2 or a.steps<40 or a.seed<0: p.error('reps >= 2, steps >= 40, seed >= 0 required')
    root=a.output; figs=root/'figures'; tables=root/'data'
    figs.mkdir(parents=True,exist_ok=True); tables.mkdir(exist_ok=True)
    font=style()
    run_topology(a,root,figs,tables,font)
    run_network(a,root,figs,tables,font)
    print(root)

if __name__=='__main__': main()