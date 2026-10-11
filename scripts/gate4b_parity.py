#!/usr/bin/env python
import json,math,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from gate3b_smoke_missing_eval import parse_simple_yaml,build_args,import_official_modules,load_checkpoint_strict
from models.probabilistic_timeperceiver import Model
def main():
 run=ROOT/'runs/timeperceiver__etth1__official_baseline__seed2025__20260701_203054';cp=next((run/'checkpoints').rglob('checkpoint.pth'));args=build_args(parse_simple_yaml(run/'resolved_config.yaml'),32,'cpu');provider,official_mod=import_official_modules();_,loader=provider(args,'test');raw=torch.load(cp,map_location='cpu');official=official_mod.Model(args).float();load_checkpoint_strict(official,cp);official.eval();mask=np.load(ROOT/'artifacts/masks/etth1/random/rate_0.3/seed_2025.npz')['mask'];idx=(list(range(args.seq_len//args.patch_len)),list(range(args.seq_len//args.patch_len,(args.seq_len+args.pred_len)//args.patch_len)))
 models={};disabled=Model(args,'official',False).float();disabled.load_state_dict(raw,strict=True);disabled.eval();models['adapter_disabled']=disabled
 for v in ('gaussian_head','student_t_head'):
  m=Model(args,v,True).float();missing,unexpected=m.load_state_dict(raw,strict=False)
  if unexpected or not all(x.startswith('probabilistic_head.') for x in missing):raise RuntimeError(f'checkpoint extension mismatch {v}')
  m.initialize_probabilistic_location();m.eval();models[v]=m
 diffs={k:0. for k in models};sums={k:[0.,0.,0] for k in models};scale_err=0.;df_err=0.;count=0;offset=0
 with torch.no_grad():
  for bi,(bx,by,bxm,bym) in enumerate(loader):
   if bi>=8:break
   bx=bx.float();by=by.float();bxm=bxm.float();bym=bym.float();mb=torch.from_numpy(mask[offset:offset+len(bx)]).float();offset+=len(bx);masked=bx*mb;dec=torch.cat([by[:,:args.label_len,:],torch.zeros_like(by[:,-args.pred_len:,:])],1);ref=official(masked,bxm,dec,bym,idx);target=by[:,-args.pred_len:,:]
   for k,m in models.items():
    value=m(masked,bxm,dec,bym,idx,mask=mb);out=value if k=='adapter_disabled' else value['loc'];diffs[k]=max(diffs[k],float((out-ref).abs().max()));d=out-target;sums[k][0]+=float((d*d).sum());sums[k][1]+=float(d.abs().sum());sums[k][2]+=d.numel()
    if k!='adapter_disabled':scale_err=max(scale_err,float((value['scale']-.5).abs().max()))
    if k=='student_t_head':df_err=max(df_err,float((value['df']-8.).abs().max()))
 results={}
 refm=json.loads((ROOT/'runs/timeperceiver__etth1__official_missing_eval_smoke__random_rate0.3__seed2025__20260709_153942/metrics.json').read_text())
 for k in models:
  mse=sums[k][0]/sums[k][2];mae=sums[k][1]/sums[k][2];rmse=math.sqrt(mse);results[k]={'forecast_max_abs_diff':diffs[k],'mse_diff':mse-refm['mse'],'mae_diff':mae-refm['mae'],'rmse_diff':rmse-refm['rmse'],'mse':mse,'mae':mae,'rmse':rmse}
 results['gaussian_head']['initial_scale_max_error']=scale_err;results['student_t_head']['initial_scale_max_error']=scale_err;results['student_t_head']['initial_df_max_error']=df_err
 passed=results['adapter_disabled']['forecast_max_abs_diff']==0 and all(results['adapter_disabled'][x]==0 for x in ('mse_diff','mae_diff','rmse_diff')) and all(results[v]['forecast_max_abs_diff']<=1e-7 and abs(results[v]['mse_diff'])<=1e-8 and abs(results[v]['mae_diff'])<=1e-8 and abs(results[v]['rmse_diff'])<=1e-8 for v in ('gaussian_head','student_t_head')) and scale_err<=1e-7 and df_err<=1e-7
 out={'status':'PASS' if passed else 'BLOCKED_PROB_HEAD_PARITY_FAILURE','evaluated_batches':8,'evaluated_samples':256,'results':results};(ROOT/'results/gate4b_parity.json').write_text(json.dumps(out,indent=2)+'\n');(ROOT/'reports/gate4b_parity.md').write_text(f"# Gate 4b Parity\n\nStatus: **{out['status']}**\n\n```json\n{json.dumps(results,indent=2)}\n```\n");print(json.dumps(out,indent=2));return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
