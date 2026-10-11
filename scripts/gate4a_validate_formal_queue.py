#!/usr/bin/env python
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];QUEUE=ROOT/'configs/queues/gate4a_minimal_adapter.yaml';PY='/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python'
def main():
 q=json.loads(QUEUE.read_text());errors=[];tasks=q.get('tasks',[])
 if q.get('logical_task_count')!=2:errors.append('logical_task_count != 2')
 if q.get('active_task_count')!=2:errors.append('active_task_count != 2')
 if {t.get('variant') for t in tasks}!={'static_adapter','missingness_aware_adapter'}:errors.append('variant set mismatch')
 if q.get('required_free_mib')!=4096:errors.append('queue required_free_mib != 4096')
 for i,t in enumerate(tasks):
  tf=ROOT/'runtime'/f'gate4a_validate_task_{i}.json';tf.parent.mkdir(exist_ok=True);tf.write_text(json.dumps(t,indent=2)+'\n');r=subprocess.run([PY,str(ROOT/'scripts/gate4a_formal_runner.py'),'--task-file',str(tf),'--validate-only'],text=True,capture_output=True);errors.extend([f"{t.get('variant')}: dry parse failed: {r.stdout.strip()}"] if r.returncode else [])
  if t.get('command')!=['gate4a_formal_runner','--task-file','{task_file}','--run-dir','{run_dir}']:errors.append(f"{t.get('variant')}: executable command mismatch")
 forbidden=('probability_head','conformal','meta_learning','test_time_adaptation','official_baseline_retrained')
 for k in forbidden:
  if q.get(k):errors.append(f'forbidden enabled: {k}')
 out={'status':'PASS' if not errors else 'BLOCKED_FORMAL_QUEUE_VALIDATION_FAILURE','queue':str(QUEUE),'logical_tasks':q.get('logical_task_count'),'active_tasks':q.get('active_task_count'),'errors':errors};(ROOT/'results/gate4a_formal_queue_validation.json').write_text(json.dumps(out,indent=2)+'\n');(ROOT/'reports/gate4a_formal_queue_validation.md').write_text(f"# Gate 4a Formal Queue Validation\n\nStatus: **{out['status']}**\n\nErrors: {errors or 'none'}\n");print(json.dumps(out,indent=2));return 0 if not errors else 2
if __name__=='__main__':raise SystemExit(main())
