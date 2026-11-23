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

# ===================== payoff helper =====================
def payoff(cp,cm,g):
    a=BASE.copy(); a[0,:,0]-=cp; a[:,0,1]-=cm; a[0,0,:]+=g
    return a

# ===================== experiment 1 (model) =====================
def run_model_experiment(a,root,figs,tables,font):
    keys=[('BA–WS','E1'),('BA–BA','E1'),('WS–WS','E1'),('BA–WS','E2'),('BA–WS','E3'),('BA–WS','E4')]; data={k:[] for k in keys}; seeds=[a.seed+r*12347 for r in range(a.reps)]
    for r,seed in enumerate(seeds):
        cache={}
        for top,case in keys:
            if top not in cache: cache[top]=setup(seed,top)
            *net,sp,sm,stream=cache[top]; rng=np.random.default_rng(stream); noise=rng.uniform(-1,1,(a.steps,len(net[0]),2)); draws=rng.random((a.steps,NP+NM,4))
            data[top,case].append(run(*net,sp.copy(),sm.copy(),CASES[case],WIDTH,noise,draws,a.alpha,.5,.5,1.,1.,1.,a.delta_p,a.delta_m))
        if (r+1)%10==0 or r+1==a.reps: print(f'[model] {r+1}/{a.reps} paired replications',flush=True)
    data={k:np.stack(v) for k,v in data.items()}; W=min(40,a.steps)
    with (tables/'trajectories_model.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['topology','case','rep','seed','step',*METRICS]);
        for (top,c),arr in data.items():
            for r,sim in enumerate(arr):
                for t,v in enumerate(sim): w.writerow([top,c,r+1,seeds[r],t,*[f'{x:.8f}' for x in v]])
    with (tables/'summary_model.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['topology','case','metric','reps','window','mean','sd','ci95_low','ci95_high']);
        for (top,c),arr in data.items():
            for k,m in enumerate(METRICS):
                v=arr[:,-W:,k].mean(1); x,h=ci(v); w.writerow([top,c,m,a.reps,W,x,v.std(ddof=1),x-h,x+h])
    with (tables/'paired_contrasts_model.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['topology','case','reference','metric','mean_difference','ci95_low','ci95_high']); ref=data[keys[0]][:,-W:,:].mean(1)
        for top,c in keys[1:]:
            for k,m in enumerate(METRICS):
                v=data[top,c][:,-W:,k].mean(1)-ref[:,k]; x,h=ci(v); w.writerow([top,c,'BA–WS/E1',m,x,x-h,x+h])
    with (tables/'payoff_matrices_model.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['case','strategy','P_mean','P_low','P_high','M_mean','M_low','M_high']);
        for c,m in CASES.items():
            for i,j,s in [(0,0,'OA'),(0,1,'OI'),(1,0,'CA'),(1,1,'CI')]:
                p,q=m[i,j]; u,v=WIDTH[i,j]; w.writerow([c,s,p,p-u,p+u,q,q-v,q+v])
    plot(data,figs,'Fig01_network_evolution',keys[:3],[k[0] for k in keys[:3]],(0,1),a.steps)
    k2=[keys[0],*keys[3:]]; labs=[LABELS[k[1]] for k in k2]
    plot(data,figs,'Fig02_incentive_evolution',k2,labs,(0,1),a.steps)
    plot(data,figs,'Fig03_coordination',k2,labs,(2,3),a.steps)
    return seeds, W

# ===================== experiment 2 (payoff sweep) =====================
def run_payoff_sweep(a,root,figs,tables,font):
    costs=[-1.5,-1.,-.5,0,.5,1.,1.5]; mc=[-1.5,-1.,-.75,-.5,0,.5,1.]; gains=[0,.3,.6,.7,.8,.9,1.,1.5,2.,2.5]
    groups={'P':[(v,0,0) for v in costs],'M':[(0,v,0) for v in mc],'G':[(0,0,v) for v in gains]}
    keys=list(dict.fromkeys(sum(groups.values(),[])+[(-.5,-.5,0),(0,0,.5)])); data={k:[] for k in keys}; seeds=[a.seed+r*12347 for r in range(a.reps)]
    for r,seed in enumerate(seeds):
        *net,sp,sm,stream=setup(seed,'BA–WS'); rng=np.random.default_rng(stream); noise=rng.uniform(-1,1,(a.steps,len(net[0]),2)); draws=rng.random((a.steps,NP+NM,4))
        for k in keys: data[k].append(run(*net,sp.copy(),sm.copy(),payoff(*k),WIDTH,noise,draws,a.alpha,.5,.5,1.,1.,1.,1.,1.))
        if (r+1)%10==0 or r+1==a.reps: print(f'[sweep] {r+1}/{a.reps} replications',flush=True)
    data={k:np.stack(v) for k,v in data.items()}; W=40; ref=data[(0,0,0)][:,-W:,:].mean(1)
    with (tables/'trajectories_sweep.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['cost_P','cost_M','joint_gain','rep','seed','step',*METRICS]);
        for k,arr in data.items():
            for r,sim in enumerate(arr):
                for t,v in enumerate(sim): w.writerow([*k,r+1,seeds[r],t,*[f'{x:.8f}' for x in v]])
    with (tables/'summary_sweep.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['cost_P','cost_M','joint_gain','metric','mean','sd','ci_low','ci_high','paired_delta','delta_ci_low','delta_ci_high']);
        for k,arr in data.items():
            for j,m in enumerate(METRICS):
                v=arr[:,-W:,j].mean(1); x,h=ci(v); d,dh=ci(v-ref[:,j]); w.writerow([*k,m,x,v.std(ddof=1),x-h,x+h,d,d-dh,d+dh])
    with (tables/'payoff_matrices_sweep.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['cost_P','cost_M','joint_gain','strategy','P_low','P_high','M_low','M_high']);
        for k in keys:
            m=payoff(*k)
            for i,j,s in [(0,0,'OA'),(0,1,'OI'),(1,0,'CA'),(1,1,'CI')]: w.writerow([*k,s,m[i,j,0]-WIDTH[i,j,0],m[i,j,0]+WIDTH[i,j,0],m[i,j,1]-WIDTH[i,j,1],m[i,j,1]+WIDTH[i,j,1]])
    colors=['#355C7D','#378D85','#898D93','#C17B49','#8878A3']; styles=['-','--','-.',':','-']; markers=['o','s','^','D','v']
    hh=np.sqrt(np.log(40)/(2*a.reps))
    for idx,(group,ks) in enumerate(groups.items(),4):
        fig,axs=plt.subplots(1,2,figsize=(7.5,3.65)); fig.subplots_adjust(left=.10,right=.98,bottom=.25,top=.80,wspace=.36)
        j={'P':0,'M':1,'G':2}[group]; sym={'P':r'\Delta c_P','M':r'\Delta c_M','G':'g'}[group]; pos='PMG'.index(group)
        selected={'P':[-1.5,-.5,0,.5,1.5],'M':[-1.5,-.75,0,.5,1.],'G':[0,.6,.7,.8,.9]}[group]
        ax=axs[0]
        for i,k in enumerate([k for k in ks if k[pos] in selected]):
            m,h=ci(data[k][:,:,j]); t=np.arange(a.steps+1)
            ax.plot(t,m,color=colors[i],ls=styles[i],marker=markers[i],markevery=max(1,a.steps//10),ms=3,mfc='white',lw=1.8,label=f'${sym}={k[pos]:g}$'); ax.fill_between(t,np.clip(m-h,0,1),np.clip(m+h,0,1),color=colors[i],alpha=.09,lw=0)
        ax.set(xlim=(0,a.steps),xlabel='Monte Carlo steps',ylabel=['Platform openness','SME adoption','Business-link collaboration'][j]); ax.legend(loc='lower left',bbox_to_anchor=(0,1.03),ncol=2,frameon=False,fontsize=8,columnspacing=.7,handlelength=1.8)
        ax=axs[1]; xx=np.array([k[pos] for k in ks]); vv=np.array([data[k][:,-W:,2].mean(1) for k in ks]); m=vv.mean(1); lo=np.maximum(0,m-hh); hi=np.minimum(1,m+hh)
        ax.errorbar(xx,m,yerr=[m-lo,hi-m],color='#355C7D',marker='o',mfc='white',lw=1.65,capsize=2.5,ms=4,label='Collaboration')
        if group=='G':
            for v in [1,2]: ax.axvline(v,color='#898D93',lw=.8,ls='--',zorder=0)
        else:
            own=np.array([data[k][:,-W:,j].mean(1).mean() for k in ks]); ax.plot(xx,own,color='#C17B49',ls='--',marker='s',mfc='white',ms=3.5,lw=1.5,label=['Platform openness','SME adoption'][j]); ax.legend(loc='best',fontsize=8,frameon=False)
        ax.set(xlabel=f'${sym}$ (normalized units)',ylabel='Final-window mean'); ax.set_xticks(xx if group!='G' else [0,.5,1,1.5,2,2.5]); ax.tick_params(axis='x',labelsize=8)
        for z,ax in enumerate(axs):
            ax.set_ylim(-.025,1.025); ax.set_yticks(np.arange(0,1.01,.2)); ax.grid(color='#D9DEE3',lw=.45,alpha=.7); ax.set_axisbelow(True); ax.text(.5,-.31,f'({"ab"[z]})',ha='center',transform=ax.transAxes,fontsize=12)
        save(fig,figs,f'Fig{idx:02d}_{group}_response')
    with (tables/'collaboration_response.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['cost_P','cost_M','joint_gain','mean','hoeffding_low','hoeffding_high','full_final_window_runs','zero_final_window_runs'])
        for k,arr in data.items():
            v=arr[:,-W:,2].mean(1); m=v.mean(); w.writerow([*k,m,max(0,m-hh),min(1,m+hh),sum(v==1),sum(v==0)])
    with (tables/'nominal_contrasts.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['case','cost_P','cost_M','joint_gain','mean_collaboration','paired_difference','t_ci_low','t_ci_high'])
        for label,k in [('baseline',(0,0,0)),('platform',(-1,0,0)),('SME',(0,-1,0)),('both',(-.5,-.5,0)),('joint_gain',(0,0,.5))]:
            v=data[k][:,-W:,2].mean(1); d,h=ci(v-ref[:,2]); w.writerow([label,*k,v.mean(),d,d-h,d+h])
    return seeds, W, groups

# ===================== main =====================
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reps',type=int,default=100); p.add_argument('--steps',type=int,default=300)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--alpha',type=float,default=.5)
    p.add_argument('--delta-p',type=float,default=1.); p.add_argument('--delta-m',type=float,default=1.)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'experiment2_results')
    a=p.parse_args()
    if a.reps<2 or a.steps<40 or a.seed<0: p.error('reps >= 2, steps >= 40, seed >= 0 required')
    root=a.output; figs=root/'figures'; tables=root/'data'
    figs.mkdir(parents=True,exist_ok=True); tables.mkdir(exist_ok=True)
    font=style()

    # ---- experiment 1 (model) ----
    seeds_m, W_m = run_model_experiment(a,root,figs,tables,font)
    (root/'parameters_model.json').write_text(json.dumps({
        'experiment':'model','reps':a.reps,'steps':a.steps,'seeds':seeds_m,'window':W_m,'alpha':a.alpha,
        'delta_p':a.delta_p,'delta_m':a.delta_m,'lambda':.5,'mu':.5,'beta_p':1,'beta_m':1,'theta':1,
        'population':[NP,NM],'initial_ratios':[.5,.5],'BA_m':[3,2],'WS_k':[6,4],'WS_rewire':.1,
        'business_probability':.1,'digital_probability':.1,'digital_weights':'U(0,1)',
        'platform_learning_weights':'symmetric U(0.5,1.5)',
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'networkx':nx.__version__,
                    'numba':numba.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},
        'font':font,
        'notes':'Stylized scenarios. Net payoff may be negative. Digital payoff differences and behavioral parameters are reconstruction assumptions, not verified original code. Paired common random numbers; independent graph and initialization per replication.'},
        indent=2,ensure_ascii=False),encoding='utf-8')

    # ---- experiment 2 (payoff sweep) ----
    seeds_s, W_s, groups = run_payoff_sweep(a,root,figs,tables,font)
    (root/'parameters_sweep.json').write_text(json.dumps({
        'experiment':'payoff_sweep','reps':a.reps,'steps':a.steps,'seed':a.seed,'seeds':seeds_s,'window':W_s,
        'alpha':a.alpha,'groups':groups,'topology':'BA–WS','population':[NP,NM],'lambda':.5,'mu':.5,
        'beta_p':1,'beta_m':1,'theta':1,'delta_p':1,'delta_m':1,'font':font,
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'networkx':nx.__version__,
                    'numba':numba.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},
        'response_CI':'pointwise 95% Hoeffding; trajectory and summary t intervals descriptive',
        'extra_contrasts':[[-.5,-.5,0],[0,0,.5]],
        'notes':'Incremental normalized costs relative to baseline; negative values are cost reductions. Joint gain added to both OA payoffs. Widths unchanged. Common random numbers. Payoffs pre-update, strategy metrics post-update. Not empirically calibrated.'},
        indent=2),encoding='utf-8')

    print(root)

if __name__=='__main__': main()