#!/usr/bin/env python
"""Validate a Gate 4a task then execute the proven adapter training path."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def load(path): return json.loads(Path(path).read_text())
def early_stopping_trace(losses,patience=3):
    best=float('inf');best_epoch=0;bad=0;stop_epoch=None
    for epoch,value in enumerate(losses,1):
        if value<best: best=value;best_epoch=epoch;bad=0
        else: bad+=1
        if bad>=patience: stop_epoch=epoch;break
    return {'best_epoch':best_epoch,'best_val':best,'stop_epoch':stop_epoch,'bad_epochs':bad}
def resolve(task):
    required={'variant','dataset','seq_len','pred_len','seed','missing_type','missing_rate','freeze_backbone','max_epochs','patience','early_stopping_metric','test_used_for_model_selection','required_free_mib','source_checkpoint','validation_mask','test_mask','config'}
    missing=sorted(required-set(task)); errors=[f'missing field: {x}' for x in missing]
    expected={'dataset':'etth1','seq_len':384,'pred_len':96,'seed':2025,'missing_type':'random','missing_rate':0.3,'freeze_backbone':True,'max_epochs':20,'patience':3,'early_stopping_metric':'validation_mse','test_used_for_model_selection':False,'required_free_mib':4096}
    for k,v in expected.items():
        if task.get(k)!=v: errors.append(f'{k}: expected {v!r}, got {task.get(k)!r}')
    if task.get('variant') not in {'static_adapter','missingness_aware_adapter'}: errors.append('invalid variant')
    for k in ('source_checkpoint','validation_mask','test_mask','config'):
        if k in task and not (ROOT/task[k]).is_file() and not Path(task[k]).is_file(): errors.append(f'missing path {k}: {task[k]}')
    for k in ('max_train_batches','max_val_batches','max_test_batches'):
        if task.get(k) is not None: errors.append(f'{k} must be null/unlimited')
    return errors
def main():
    p=argparse.ArgumentParser();p.add_argument('--task-file');p.add_argument('--run-dir');p.add_argument('--validate-only',action='store_true');p.add_argument('--test-early-stopping',action='store_true');a=p.parse_args()
    if a.test_early_stopping:
        declining=early_stopping_trace([1.0,0.9,0.91,0.92,0.93]);improving=early_stopping_trace([1.0,0.9,0.8,0.7,0.6]);ok=declining=={'best_epoch':2,'best_val':0.9,'stop_epoch':5,'bad_epochs':3} and improving['stop_epoch'] is None;print(json.dumps({'status':'PASS' if ok else 'FAIL','artificial_sequence':declining,'continuous_improvement':improving},indent=2));return 0 if ok else 2
    if not a.task_file: raise SystemExit('--task-file required')
    task=load(a.task_file);errors=resolve(task)
    if a.validate_only: print(json.dumps({'status':'PASS' if not errors else 'FAIL','task':task,'errors':errors},indent=2));return 0 if not errors else 2
    if errors: print(json.dumps({'status':'CONFIG_ERROR','errors':errors},indent=2));return 2
    if not a.run_dir: raise SystemExit('--run-dir required for execution')
    cmd=[sys.executable,'-u',str(ROOT/'scripts/gate4a_train_adapter.py'),'--variant',task['variant'],'--run-dir',a.run_dir,'--device','cuda']
    return subprocess.call(cmd,env=os.environ.copy(),cwd=str(ROOT))
if __name__=='__main__': raise SystemExit(main())
