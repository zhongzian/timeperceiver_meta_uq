import math
import numpy as np
import torch
from scipy.special import ndtri,stdtrit

LEVELS=(0.5,0.8,0.9,0.95)
def distribution(variant,loc,scale,df=None):
    return torch.distributions.Normal(loc,scale) if variant=='gaussian_head' else torch.distributions.StudentT(df,loc,scale)
def nll_elements(variant,loc,scale,target,df=None):return -distribution(variant,loc,scale,df).log_prob(target)
def central_intervals(variant,loc,scale,df=None):
    out={}
    for level in LEVELS:
        q=(1+level)/2
        if variant=='gaussian_head':z=torch.as_tensor(float(ndtri(q)),device=loc.device,dtype=loc.dtype)
        else:z=torch.from_numpy(stdtrit(df.detach().cpu().numpy(),q)).to(loc.device,loc.dtype)
        out[level]=(loc-z*scale,loc+z*scale)
    return out
def interval_metrics(target,intervals):
    out={};aces=[];widths=[];scores=[]
    for level,(lo,hi) in intervals.items():
        alpha=1-level;inside=((target>=lo)&(target<=hi)).float();width=hi-lo;score=width+(2/alpha)*(lo-target)*(target<lo)+(2/alpha)*(target-hi)*(target>hi);tag=str(int(level*100));picp=float(inside.mean());mpiw=float(width.mean());ival=float(score.mean());out.update({f'picp_{tag}':picp,f'mpiw_{tag}':mpiw,f'coverage_gap_{tag}':picp-level,f'absolute_coverage_error_{tag}':abs(picp-level),f'interval_score_{tag}':ival});aces.append(abs(picp-level));widths.append(mpiw);scores.append(ival)
    out.update({'mean_absolute_coverage_error':float(np.mean(aces)),'mean_interval_width':float(np.mean(widths)),'mean_interval_score':float(np.mean(scores))});return out
def empirical_crps(variant,loc,scale,target,df=None,samples=64,seed=52025,chunk_elements=65536):
    lf=loc.reshape(-1);sf=scale.reshape(-1);yf=target.reshape(-1);dfv=df.reshape(-1) if df is not None else None;total=0.;count=0
    devices=[loc.device.index] if loc.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for start in range(0,lf.numel(),chunk_elements):
            l=lf[start:start+chunk_elements];s=sf[start:start+chunk_elements];y=yf[start:start+chunk_elements];d=None if dfv is None else dfv[start:start+chunk_elements];dist=distribution(variant,l,s,d);x=dist.sample((samples,));first=(x-y).abs().mean(0);xs=x.sort(0).values;coeff=(2*torch.arange(1,samples+1,device=x.device,dtype=x.dtype)-samples-1).view(-1,1);pair_half=(coeff*xs).sum(0)/(samples*samples);crps=first-pair_half;total+=float(crps.sum());count+=crps.numel()
    return total/count
