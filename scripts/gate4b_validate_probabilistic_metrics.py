#!/usr/bin/env python
import json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from gate4b_probabilistic_metrics import nll_elements,central_intervals,empirical_crps
def main():
 loc=torch.tensor([0.,.2]);scale=torch.tensor([.5,1.1]);y=torch.tensor([.1,-.3]);df=torch.tensor([8.,8.])
 gn=nll_elements('gaussian_head',loc,scale,y);gm=torch.log(scale)+.5*math.log(2*math.pi)+.5*((y-loc)/scale)**2
 tn=nll_elements('student_t_head',loc,scale,y,df);z=(y-loc)/scale;tm=torch.log(scale)+.5*torch.log(df*math.pi)+torch.lgamma(df/2)-torch.lgamma((df+1)/2)+((df+1)/2)*torch.log1p(z*z/df)
 one=torch.zeros(1);unit=torch.ones(1);gi=central_intervals('gaussian_head',one,unit)[.9];ti=central_intervals('student_t_head',one,unit,torch.full((1,),8.))[.9]
 nested=True
 ints=central_intervals('student_t_head',one,unit,torch.full((1,),8.));prev=None
 for level in (.5,.8,.9,.95):
  lo,hi=ints[level]
  if prev is not None:nested&=bool(lo<=prev[0] and hi>=prev[1])
  nested&=bool(hi>lo);prev=(lo,hi)
 torch.manual_seed(123);n=200000;target=torch.zeros(n);loc0=torch.zeros(n);scale1=torch.ones(n);mc=empirical_crps('gaussian_head',loc0,scale1,target,samples=256,seed=777,chunk_elements=50000);analytic=2/math.sqrt(2*math.pi)-1/math.sqrt(math.pi)
 checks={'gaussian_nll_max_diff':float((gn-gm).abs().max()),'student_t_nll_max_diff':float((tn-tm).abs().max()),'gaussian_90_upper':float(gi[1]),'student_t_df8_90_upper':float(ti[1]),'student_interval_wider':bool(ti[1]>gi[1]),'intervals_nested_positive':nested,'crps_mc':mc,'crps_gaussian_analytic':analytic,'crps_abs_error':abs(mc-analytic)}
 passed=checks['gaussian_nll_max_diff']<1e-6 and checks['student_t_nll_max_diff']<1e-6 and abs(checks['gaussian_90_upper']-1.6448536269)<1e-6 and checks['student_interval_wider'] and nested and checks['crps_abs_error']<.01
 out={'status':'PASS' if passed else 'BLOCKED_PROBABILISTIC_METRIC_VALIDATION_FAILURE','checks':checks,'formal_crps_method':'empirical_monte_carlo','formal_crps_samples':64,'formal_crps_seed':52025};(ROOT/'results/gate4b_metric_validation.json').write_text(json.dumps(out,indent=2)+'\n');(ROOT/'reports/gate4b_metric_validation.md').write_text(f"# Gate 4b Metric Validation\n\nStatus: **{out['status']}**\n\n```json\n{json.dumps(checks,indent=2)}\n```\n");print(json.dumps(out,indent=2));return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
