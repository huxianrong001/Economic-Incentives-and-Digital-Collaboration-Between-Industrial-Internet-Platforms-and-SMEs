#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Combined experiments: topology/incentive benchmark, policy portfolio, and robustness checks."""
import argparse, csv, json, platform, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
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

def setup(seed,topology,np_dim=None,nm_dim=None):
    NP_=NP if np_dim is None else np_dim
    NM_=NM if nm_dim is None else nm_dim
    streams=np.random.SeedSequence(seed).spawn(5)
    rng=np.random.default_rng(streams[0]); business=rng.random((NP_,NM_))<.1; digital=rng.random((NP_,NM_))<.1
    for a in (business,digital):
        for i in np.flatnonzero(~a.any(axis=1)): a[i,rng.integers(NM_)]=True
        for j in np.flatnonzero(~a.any(axis=0)): a[rng.integers(NP_),j]=True
    weights=rng.random((NP_,NM_))*digital
    bi,bj=np.nonzero(business); dw_p=weights.sum(1)/digital.sum(1); dw_m=weights.sum(0)/digital.sum(0)
    graphs=[]
    for n,kind,m,k,s in zip((NP_,NM_),TOPO[topology],(3,2),(6,4),streams[1:3]):
        r=np.random.default_rng(s); gseed=int(r.integers(2**31-1)); g=nx.barabasi_albert_graph(n,m,seed=gseed) if kind=='BA' else nx.watts_strogatz_graph(n,k,.1,seed=gseed)
        graphs.append(pack(g,r))
    r=np.random.default_rng(streams[3]); sp=np.zeros(NP_,np.int64); sm=np.zeros(NM_,np.int64)
    sp[r.choice(NP_,NP_//2,False)]=1; sm[r.choice(NM_,NM_//2,False)]=1
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
        if (sp.min()==sp.max()) and (sm.min()==sm.max()):
            out[t+1,:4]=out[t,:4]; continue
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

# ===================== experiment A: topology + incentive =====================
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

# ===================== experiment B: policy portfolio =====================
G=[0.,.6,.7]; B=[.05,.1,.2,.3,.4,.6,.8]; PB=[.1,.2,.4]
PURE=[(1.,0.,0.),(0.,1.,0.),(0.,0.,1.)]; Q=[(i/4,j/4,(4-i-j)/4) for i in range(5) for j in range(5-i)]
KEYS=[(g,0.,0.,0.,0.) for g in G]+[(g,b,*q) for g in G for b in B for q in PURE]+[(g,b,*q) for g in G[1:] for b in PB for q in Q if q not in PURE]

def worker(args):
    seed,steps=args; *net,sp,sm,stream=setup(seed,'BA–WS'); rng=np.random.default_rng(stream); noise=rng.uniform(-1,1,(steps,len(net[0]),2)); draws=rng.random((steps,NP+NM,4)); out=[]
    for g,b,p,m,d in KEYS:
        mat=BASE.copy(); mat[0,0,:]+=g; mat[0,:,0]+=5*b*p; mat[:,0,1]+=1.25*b*m
        out.append(run(*net,sp.copy(),sm.copy(),mat,WIDTH,noise,draws,.5+b*d,.5,.5,1.,1.,1.,1.,1.))
    return np.stack(out)

def run_policy(a,root,figs,tables):
    seeds=(a.seed+np.arange(a.reps)*12347)
    if a.reuse and (tables/'trajectories_policy.npz').exists():
        z=np.load(tables/'trajectories_policy.npz'); assert np.array_equal(z['keys'],KEYS) and np.array_equal(z['seeds'],seeds) and z['trajectories'].shape[2]==a.steps+1; arr=z['trajectories']
    else:
        vals=[]
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for i,v in enumerate(pool.map(worker,[(int(s),a.steps) for s in seeds])):
                vals.append(v)
                if (i+1)%10==0: print(f'[policy] {i+1}/{a.reps}',flush=True)
        arr=np.stack(vals,axis=1); np.savez_compressed(tables/'trajectories_policy.npz',keys=KEYS,seeds=seeds,trajectories=arr)
    data=dict(zip(KEYS,arr)); rows={}; w=40
    for k,v in data.items():
        g,b,p,m,d=k; final=v[:,-w:,:].mean(1); base=data[g,0.,0.,0.,0.][:,-w:,2].mean(1); gain=final[:,2]-base; mu,h=ci(gain); cm,ch=ci(final[:,2]); spend=b*(p*v[:,-w-1:-1,0]+m*v[:,-w-1:-1,1]+d); cost=spend.mean(); total=b*(p*v[:,:-1,0]+m*v[:,:-1,1]+d)
        rows[k]=dict(g=g,budget=b,share_P=p,share_M=m,share_D=d,openness=final[:,0].mean(),adoption=final[:,1].mean(),collaboration=cm,sd=final[:,2].std(ddof=1),ci_low=max(0,cm-ch),ci_high=min(1,cm+ch),paired_gain=mu,paired_ci_low=mu-h,paired_ci_high=mu+h,actual_spend=cost,cumulative_actual_spend=total.sum(1).mean(),CE_reserved=mu/b if b else np.nan,CE_actual=mu/cost if cost>0 else np.nan,CE_full_horizon=mu/total.sum(1).mean() if total.sum(1).mean()>0 else np.nan)
    with (tables/'summary_policy.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(next(iter(rows.values())))); writer.writeheader(); writer.writerows(rows.values())
    colors=['#355C7D','#C17B49','#378D85']; labels=['Platform subsidy','SME subsidy','Digital support']; marks=['o','s','^']; lines=['-','--','-.']
    def decorate(ax,i,g,y):
        ax.set(xlabel='Reserved budget per agent',ylabel=y); ax.grid(color='#D9DEE3',lw=.45,alpha=.7); ax.set_axisbelow(True); ax.text(.5,-.35,f'({chr(97+i)})  $g={g:g}$',transform=ax.transAxes,ha='center')
    f,axes=plt.subplots(1,3,figsize=(11,3.5)); f.subplots_adjust(left=.075,right=.98,bottom=.27,top=.83,wspace=.36)
    for j,g in enumerate(G):
        ax=axes[j]; base=rows[g,0.,0.,0.,0.]
        for i,q in enumerate(PURE):
            rr=[base]+[rows[g,b,*q] for b in B]; x=[0,*B]; ax.plot(x,[r['collaboration'] for r in rr],color=colors[i],ls=lines[i],marker=marks[i],mfc='white',ms=4,label=labels[i]); ax.fill_between(x,[r['ci_low'] for r in rr],[r['ci_high'] for r in rr],color=colors[i],alpha=.1)
        decorate(ax,j,g,'Final-window collaboration'); ax.set_ylim(-.025,1.025)
    f.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False); save(f,figs,'Fig09_policy_response')
    f,axes=plt.subplots(2,3,figsize=(11,7)); f.subplots_adjust(left=.08,right=.98,bottom=.13,top=.88,wspace=.42,hspace=.65)
    for r,metric in enumerate(['CE_reserved','CE_actual']):
        for j,g in enumerate(G):
            ax=axes[r,j]
            for i,q in enumerate(PURE): ax.plot(B,[rows[g,b,*q][metric] for b in B],color=colors[i],ls=lines[i],marker=marks[i],mfc='white',ms=4,label=labels[i])
            ax.axhline(0,color='.6',lw=.6); decorate(ax,r*3+j,g,'Gain / reserved budget' if r==0 else 'Gain / actual spending')
    f.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False); save(f,figs,'Fig10_budget_efficiency')
    f,axes=plt.subplots(2,3,figsize=(10,6.8)); f.subplots_adjust(left=.04,right=.87,bottom=.1,top=.96,wspace=.18,hspace=.35)
    pts=np.array([(q[1]+.5*q[2],np.sqrt(3)/2*q[2]) for q in Q])
    for r,g in enumerate(G[1:]):
        for j,b in enumerate(PB):
            ax=axes[r,j]; vals=[rows[g,b,*q]['collaboration'] for q in Q]; ax.plot([0,1,.5,0],[0,0,np.sqrt(3)/2,0],color='.6',lw=.9); sc=ax.scatter(*pts.T,c=vals,cmap='viridis',vmin=0,vmax=1,s=100,edgecolors='white',linewidths=.7); ax.text(-.04,-.05,'P',ha='center'); ax.text(1.04,-.05,'M',ha='center'); ax.text(.5,.91,'D',ha='center'); ax.set(xlim=(-.12,1.12),ylim=(-.12,1),aspect='equal'); ax.axis('off'); ax.text(.5,-.08,f'({chr(97+r*3+j)})  $g={g:g}, B={b:g}$',transform=ax.transAxes,ha='center')
    cax=f.add_axes([.9,.22,.018,.58]); f.colorbar(sc,cax=cax,label='Final-window collaboration'); save(f,figs,'Fig11_portfolio')
    (root/'parameters_policy.json').write_text(json.dumps(dict(experiment='policy',reps=a.reps,steps=a.steps,seed=a.seed,window=w,keys=KEYS,budget_definition='reserved per agent per MCS; digital conversion cost 1; conditional payouts on pre-update state'),indent=2))

# ===================== experiment C: robustness checks =====================
ROB_BASE=(1.,'Hybrid',50,200,.5)
ROB_GROUPS=[('Payoff uncertainty',[0.,.5,1.,1.5,2.],0),
            ('Updating rules',['Fermi','Weighted','Hybrid'],1),
            ('Population size',[(25,100),(50,200),(75,300)],2),
            ('Initial participation',[.2,.35,.5,.65,.8],4)]
def rob_condition(idx,val):
    c=list(ROB_BASE)
    if idx==2: c[2:4]=val
    else: c[idx]=val
    return tuple(c)
ROB_CONDITIONS=list(dict.fromkeys(rob_condition(idx,v) for _,vs,idx in ROB_GROUPS for v in vs))
ROB_SCENARIOS=[(0.,.5,'None'),(.7,.5,'None'),(0.,.75,'None'),(.7,.75,'None'),(.7,.5,'P'),(.7,.5,'M'),(.7,.6,'D')]
ROB_CONTRASTS=['Joint benefit','Empowerment interaction','SME vs platform','SME vs digital']

def rob_worker(args):
    seed,steps,window=args; output=[]; cache={}
    for width,rule,npop,nm,initial in ROB_CONDITIONS:
        if (npop,nm) not in cache:
            *net,sp,sm,stream=setup(seed,'BA–WS',npop,nm)
            rng=np.random.default_rng(stream)
            noise=rng.uniform(-1,1,(steps,len(net[0]),2)); draws=rng.random((steps,npop+nm,4))
            r=np.random.default_rng(np.random.SeedSequence([seed,987]))
            orders=[np.r_[r.permutation(np.flatnonzero(s)),r.permutation(np.flatnonzero(1-s))] for s in (sp,sm)]
            cache[npop,nm]=net,noise,draws,orders
        net,noise,draws,orders=cache[npop,nm]
        sp=np.zeros(npop,np.int64); sm=np.zeros(nm,np.int64)
        for s,order in zip((sp,sm),orders): s[order[:round(initial*len(s))]]=1
        lam,mu={'Fermi':(0.,1.),'Weighted':(1.,1.),'Hybrid':(.5,.5)}[rule]
        results=[]
        for g,alpha,policy in ROB_SCENARIOS:
            mat=BASE.copy(); mat[0,0,:]+=g
            if policy=='P': mat[0,:,0]+=.1*(npop+nm)/npop
            if policy=='M': mat[:,0,1]+=.1*(npop+nm)/nm
            out=run(*net,sp.copy(),sm.copy(),mat,WIDTH*width,noise,draws,alpha,lam,mu,1.,1.,1.,1.,1.)
            results.append(out[-window:,:].mean(0))
        output.append(results)
    return np.array(output)

def run_robustness(a,root,figs,tables,font):
    seeds=[a.seed+i*12347 for i in range(a.reps)]; vals=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,v in enumerate(pool.map(rob_worker,[(s,a.steps,a.window) for s in seeds])):
            vals.append(v)
            if (i+1)%10==0 or i+1==a.reps: print(f'[robustness] {i+1}/{a.reps}',flush=True)
    arr=np.stack(vals,axis=1)
    np.savez_compressed(tables/'final_windows_robustness.npz',values=arr,seeds=seeds,
                        conditions=np.array(ROB_CONDITIONS,dtype=str),
                        scenarios=np.array(ROB_SCENARIOS,dtype=str),metrics=METRICS)
    c=arr[:,:,:,2]; effects=np.stack([c[:,:,1]-c[:,:,0],(c[:,:,3]-c[:,:,1])-(c[:,:,2]-c[:,:,0]),c[:,:,5]-c[:,:,4],c[:,:,5]-c[:,:,6]],axis=-1)
    fields=['width','rule','NP','NM','initial']
    with (tables/'replications_robustness.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow([*fields,'seed','scenario',*METRICS]);
        for i,cond in enumerate(ROB_CONDITIONS):
            for j,seed in enumerate(seeds):
                for k in range(7): w.writerow([*cond,seed,f'R{k+1}',*arr[i,j,k]])
    with (tables/'summary_robustness.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow([*fields,'scenario','g','alpha','policy','metric','mean','sd','ci_low','ci_high'])
        for i,cond in enumerate(ROB_CONDITIONS):
            for k,sc in enumerate(ROB_SCENARIOS):
                for l,metric in enumerate(METRICS):
                    v=arr[i,:,k,l]; z,h=ci(v); w.writerow([*cond,f'R{k+1}',*sc,metric,z,v.std(ddof=1),z-h,z+h])
    with (tables/'paired_contrasts_robustness.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow([*fields,'contrast','mean','sd','ci_low','ci_high','positive_fraction','direction'])
        for i,cond in enumerate(ROB_CONDITIONS):
            for k,name in enumerate(ROB_CONTRASTS):
                v=effects[i,:,k]; z,h=ci(v); w.writerow([*cond,name,z,v.std(ddof=1),z-h,z+h,(v>0).mean(),'positive' if z-h>0 else 'negative' if z+h<0 else 'inconclusive'])
    fig,axes=plt.subplots(2,2,figsize=(8.3,6.7)); fig.subplots_adjust(left=.12,right=.98,bottom=.16,top=.87,wspace=.30,hspace=.65)
    for k,(ax,(label,values,idx)) in enumerate(zip(axes.flat,ROB_GROUPS)):
        ids=[ROB_CONDITIONS.index(rob_condition(idx,v)) for v in values]; x=np.arange(len(ids))
        for j,name in enumerate(ROB_CONTRASTS):
            z,h=ci(effects[ids,:,j].T); ax.errorbar(x+(j-1.5)*.055,z,yerr=h,color=COLORS[j],marker=MARKERS[j],ls=STYLES[j],lw=1.25,ms=4,mfc='white',capsize=2,label=name)
        ax.axhline(0,color='.45',lw=.8); ax.grid(axis='y',color='#D9DEE3',lw=.5); ax.set_axisbelow(True); ax.set_xticks(x,[f'{v[0]}/{v[1]}' if isinstance(v,tuple) else str(v) for v in values]); ax.set_xlabel(label); ax.set_ylabel('Paired collaboration difference'); ax.text(.5,-.34,f'({chr(97+k)})',transform=ax.transAxes,ha='center',fontsize=12)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,.99)); save(fig,figs,'Fig12_robustness')
    (root/'parameters_robustness.json').write_text(json.dumps(dict(
        experiment='robustness',reps=a.reps,steps=a.steps,window=a.window,seeds=seeds,
        conditions=ROB_CONDITIONS,scenarios=ROB_SCENARIOS,contrasts=ROB_CONTRASTS,baseline=ROB_BASE,
        model_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=platform.python_version(),numpy=np.__version__,networkx=nx.__version__,
        numba=numba.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,
        font=font,ci='unadjusted paired t intervals; exploratory',
        notes='Weighted means platform weighted Fermi, SME uniform Fermi; fixed graph density across sizes; nested initial sets; no mutation; final-window metrics only'),indent=2),encoding='utf-8')

# ===================== main =====================
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reps',type=int,default=100)
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--alpha',type=float,default=.5)
    p.add_argument('--delta-p',type=float,default=1.)
    p.add_argument('--delta-m',type=float,default=1.)
    p.add_argument('--window',type=int,default=40)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--reuse',action='store_true')
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'experiment5_results')
    a=p.parse_args()
    if a.reps<2 or a.steps<40 or a.seed<0 or not 1<=a.window<=a.steps or a.workers<1:
        p.error('Invalid replication, steps, window, worker or seed setting')
    root=a.output; figs=root/'figures'; tables=root/'data'
    figs.mkdir(parents=True,exist_ok=True); tables.mkdir(exist_ok=True)
    font=style()
    run_topology(a,root,figs,tables,font)
    run_policy(a,root,figs,tables)
    run_robustness(a,root,figs,tables,font)
    print(root)

if __name__=='__main__':
    main()