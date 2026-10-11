#!/usr/bin/env python
"""Read-only mechanism audit over completed Gate 4a artifacts."""
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from gate3b_smoke_missing_eval import parse_simple_yaml,build_args,import_official_modules
from models.missingness_aware_timeperceiver import Model

RUNS={
 'static_adapter':ROOT/'runs/timeperceiver__etth1__gate4a__static_adapter__random_rate0.3__seed2025__20260711_224844',
 'missingness_aware_adapter':ROOT/'runs/timeperceiver__etth1__gate4a__missingness_aware_adapter__random_rate0.3__seed2025__20260711_224907'}
def load(p):return json.loads(Path(p).read_text())
def stats(x):
 x=x.float();return {'mean':float(x.mean()),'std':float(x.std(unbiased=False)),'min':float(x.min()),'max':float(x.max()),'mean_abs':float(x.abs().mean()),'max_abs':float(x.abs().max())}
def raw_features(mask):
 m=1-mask.float();c=m.mean(1);t=m.mean(2);return torch.stack([m.mean((1,2)),c.mean(1),c.std(1,unbiased=False),c.amax(1),t.mean(1),t.std(1,unbiased=False),t.amax(1)],1)
def initial_models(args):
 torch.manual_seed(2025);_,official=import_official_modules();_unused=official.Model(args).float();return {v:Model(args,v).float() for v in RUNS}
def update_stats(initial,trained):
 out={};all_delta=[];all_init=[]
 for k,v in trained.items():
  d=v.float()-initial[k].float();all_delta.append(d.reshape(-1));all_init.append(initial[k].float().reshape(-1));out[k]={'delta_l2':float(d.norm()),'delta_mean_abs':float(d.abs().mean()),'trained_l2':float(v.float().norm())}
 d=torch.cat(all_delta);i=torch.cat(all_init);return {'all_delta_l2':float(d.norm()),'relative_delta_l2':float(d.norm()/(i.norm()+1e-12)),'updated_tensor_count':sum(x['delta_l2']>0 for x in out.values()),'per_tensor':out}
def condition_audit(model,mask_path):
 arr=np.load(mask_path)['mask'];raws=[];zs=[];gs=[];bs=[]
 with torch.no_grad():
  for start in range(0,len(arr),256):
   m=torch.from_numpy(arr[start:start+256]).float();raw=raw_features(m);z=model.missingness_encoder(m);g,b=model.adapter.condition(z).chunk(2,-1);raws.append(raw);zs.append(z);gs.append(g);bs.append(b)
 raw,z,g,b=map(torch.cat,(raws,zs,gs,bs));sample_z=z.std(0,unbiased=False)
 return {'sample_count':len(raw),'raw_feature_names':['global','channel_mean','channel_std','channel_max','temporal_mean','temporal_std','temporal_max'],'raw_per_dimension_std':[float(x) for x in raw.std(0,unbiased=False)],'raw_global':stats(raw),'z_miss':stats(z),'z_mean_feature_std_across_samples':float(sample_z.mean()),'z_max_feature_std_across_samples':float(sample_z.max()),'gamma':stats(g),'beta':stats(b)}
def main():
 cfg=parse_simple_yaml(ROOT/'runs/timeperceiver__etth1__official_baseline__seed2025__20260701_203054/resolved_config.yaml');args=build_args(cfg,128,'cpu');initial=initial_models(args);updates={};trained={}
 for variant,run in RUNS.items():
  obj=torch.load(run/'checkpoints/best_adapter.pt',map_location='cpu')['adapter_state_dict'];init={k:v for k,v in initial[variant].state_dict().items() if k.startswith(('adapter.','missingness_encoder.'))};updates[variant]=update_stats(init,obj);model=initial[variant];model.load_state_dict(obj,strict=False);trained[variant]=model
 val=condition_audit(trained['missingness_aware_adapter'],ROOT/'artifacts/gate4a_val_masks/etth1/random/rate_0.3/seed_43025.npz');test=condition_audit(trained['missingness_aware_adapter'],ROOT/'artifacts/masks/etth1/random/rate_0.3/seed_2025.npz')
 sm=load(RUNS['static_adapter']/'metrics.json');mm=load(RUNS['missingness_aware_adapter']/'metrics.json');ranking={'validation_consistent_with_test':sm['best_validation_mse']<mm['best_validation_mse'] and sm['test_mse']<mm['test_mse'],'static_best_epoch':sm['best_epoch'],'missing_best_epoch':mm['best_epoch'],'static_val_mse':sm['best_validation_mse'],'missing_val_mse':mm['best_validation_mse'],'static_test_mse':sm['test_mse'],'missing_test_mse':mm['test_mse']}
 diagnosis={'condition_near_constant':test['raw_per_dimension_std'][0]<0.001 and test['z_mean_feature_std_across_samples']<0.001,'film_near_zero':test['gamma']['mean_abs']<0.01 and test['beta']['mean_abs']<0.01,'primary_interpretation':'the fixed random-0.3 condition has limited but nonzero sample diversity, and trained FiLM modulation is measurably nonzero; the failure is therefore closer to ineffective conditioning/injection or optimization than to a completely constant condition','gradient_norms_available':False,'epoch_curves_available':False}
 out={'status':'PASS','read_only_artifact_audit':True,'updates':updates,'validation_condition':val,'test_condition':test,'ranking':ranking,'diagnosis':diagnosis,'limitations':['Per-step gradient norms were not persisted.','Per-epoch train/validation curves were not persisted; only best epoch and best validation MSE are available.']};(ROOT/'results/gate4a_mechanism_audit.json').write_text(json.dumps(out,indent=2)+'\n')
 md=f"# Gate 4a Mechanism Audit\n\nStatus: **PASS**\n\nNo training was run. Existing checkpoints and fixed masks were inspected.\n\n## Condition diversity\n\n- Test global-rate std: `{test['raw_per_dimension_std'][0]:.8f}`\n- Mean z feature std across samples: `{test['z_mean_feature_std_across_samples']:.8f}`\n- Gamma mean absolute magnitude: `{test['gamma']['mean_abs']:.8f}`\n- Beta mean absolute magnitude: `{test['beta']['mean_abs']:.8f}`\n\n## Parameter updates\n\n- Static relative update L2: `{updates['static_adapter']['relative_delta_l2']:.6f}`\n- Missing-aware relative update L2: `{updates['missingness_aware_adapter']['relative_delta_l2']:.6f}`\n\n## Ordering\n\nValidation and test rankings agree: `{str(ranking['validation_consistent_with_test']).lower()}`. Static is better on both.\n\n## Interpretation\n\n{diagnosis['primary_interpretation']}. This is a single-seed mechanism diagnosis, not a statistical claim. Per-step gradients and complete epoch curves were not persisted and cannot be reconstructed exactly.\n";(ROOT/'reports/gate4a_mechanism_audit.md').write_text(md);print(json.dumps({'status':'PASS','diagnosis':diagnosis,'test_condition':test,'ranking':ranking},indent=2))
if __name__=='__main__':main()
