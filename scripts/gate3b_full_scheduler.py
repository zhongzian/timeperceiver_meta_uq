#!/usr/bin/env python
"""Queue-specific resumable Gate 3b-full scheduler."""
import json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; Q=ROOT/"configs/queues/gate3b_full_missing_eval.yaml"; PY=Path("/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python")
def load(p): return json.loads(Path(p).read_text())
def free():
 try:
  out=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],text=True)
  return sorted([(int(x.split(',')[1]),int(x.split(',')[0])) for x in out.splitlines()],reverse=True)
 except Exception:return []
def main():
 q=load(Q)
 for i,t in enumerate(q["tasks"]):
  # completed logical keys are reused across scheduler restarts
  pat=f"timeperceiver__{t['dataset']}__official_missing_eval_full__{t['missing_type']}_rate{t['target_missing_rate']}__seed{t['seed']}__*/status.json"
  if any(load(p).get("status")=="SUCCESS" for p in ROOT.glob("runs/"+pat)): continue
  need=q["required_free_mib"][t["dataset"]]
  while True:
   cards=[x for x in free() if x[0]>=need]
   if cards: break
   print(f"WAITING_RESOURCE task={i+1}/{len(q['tasks'])} need={need}",flush=True); time.sleep(60)
  gpu=cards[0][1]; ts=datetime.now().strftime("%Y%m%d_%H%M%S")
  rid=f"timeperceiver__{t['dataset']}__official_missing_eval_full__{t['missing_type']}_rate{t['target_missing_rate']}__seed{t['seed']}__{ts}"
  tf=ROOT/"runtime"/f"gate3b_task_{i:02d}.json"; tf.parent.mkdir(exist_ok=True); tf.write_text(json.dumps(t,indent=2)+"\n")
  env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu)
  cmd=[str(PY),str(ROOT/"scripts/gate3b_official_missing_eval.py"),"--task-file",str(tf),"--run-dir",str(ROOT/"runs"/rid),"--device","cuda"]
  print(f"START {rid} gpu={gpu} free={cards[0][0]}",flush=True)
  rc=subprocess.call(cmd,env=env); print(f"END {rid} rc={rc}",flush=True)
  if rc: time.sleep(120)
 print("QUEUE_COMPLETE",flush=True)
if __name__=="__main__": main()
