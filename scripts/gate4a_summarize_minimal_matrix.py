#!/usr/bin/env python
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads(Path(p).read_text())
def latest(variant):
    rows=[]
    for p in (ROOT/'runs').glob(f'timeperceiver__etth1__gate4a__{variant}__random_rate0.3__seed2025__*/status.json'):
        try:
            s=load(p);m=load(p.parent/'metrics.json')
            if s.get('status')=='SUCCESS' and not m.get('smoke_only'):rows.append((p.parent,m,s))
        except Exception:pass
    if not rows:raise RuntimeError(f'no formal SUCCESS run for {variant}')
    return sorted(rows,key=lambda x:x[0].name)[-1]
def main():
    with open(ROOT/'reports/gate3b_full_results.csv',newline='') as f: base=next(r for r in csv.DictReader(f) if r['dataset']=='etth1' and r['missing_type']=='random' and float(r['target_missing_rate'])==0.3 and int(r['seed'])==2025)
    official={'variant':'official_baseline','experiment_id':base['experiment_id'],'best_epoch':'NA','best_validation_mse':'NA','test_mse':float(base['mse']),'test_mae':float(base['mae']),'test_rmse':float(base['rmse']),'trainable_parameter_count':0,'total_parameter_count':None,'parameter_overhead_ratio':0.0,'run_dir':base['run_dir']}
    rows=[official]
    for v in ('static_adapter','missingness_aware_adapter'):
        run,m,s=latest(v);backbone=int(m['total_parameter_count'])-int(m['trainable_parameter_count']);rows.append({'variant':v,'experiment_id':run.name,'best_epoch':m['best_epoch'],'best_validation_mse':m['best_validation_mse'],'test_mse':m['test_mse'],'test_mae':m['test_mae'],'test_rmse':m['test_rmse'],'trainable_parameter_count':m['trainable_parameter_count'],'total_parameter_count':m['total_parameter_count'],'parameter_overhead_ratio':m['trainable_parameter_count']/backbone,'epochs_completed':m['epochs_completed'],'early_stopped':m['early_stopped'],'peak_gpu_memory_mib':m['peak_gpu_memory_mib'],'training_time_seconds':m['elapsed_seconds'],'run_dir':str(run)})
    fields=sorted({k for r in rows for k in r});
    with open(ROOT/'reports/gate4a_minimal_results.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    pfields=['variant','trainable_parameter_count','total_parameter_count','parameter_overhead_ratio'];
    with open(ROOT/'reports/gate4a_parameter_comparison.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=pfields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    md='# Gate 4a Minimal Results\n\nSingle seed (`2025`); no statistical-significance claim.\n\n| Variant | Best epoch | Val MSE | Test MSE | Test MAE | Test RMSE | Trainable params | Overhead |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in rows:md+=f"| {r['variant']} | {r['best_epoch']} | {r['best_validation_mse']} | {r['test_mse']:.6f} | {r['test_mae']:.6f} | {r['test_rmse']:.6f} | {r['trainable_parameter_count']} | {r['parameter_overhead_ratio']:.4%} |\n"
    (ROOT/'reports/gate4a_minimal_results.md').write_text(md);(ROOT/'reports/gate4a_parameter_comparison.md').write_text(md);print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
