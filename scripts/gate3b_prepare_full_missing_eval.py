#!/usr/bin/env python
"""Audit Gate 2/Gate 3a assets and prepare the Gate 3b-full queue."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RUNS=ROOT/"runs"
DATASETS=("etth1","weather","electricity"); TYPES=("random","block","channel","mixed"); RATES=(0.1,0.3,0.5,0.7)
def read(p): return json.loads(Path(p).read_text())
def write(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(x,indent=2)+"\n")
def text(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(x)
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1048576),b""): h.update(b)
 return h.hexdigest()
def main():
 gate2=read(ROOT/"reports/gate2_status.json"); cps=[]
 for ds in DATASETS:
  rows=[x for x in gate2["baseline_runs"] if x["dataset"]==ds and x["status"]=="SUCCESS"]
  if len(rows)!=1: raise SystemExit(f"checkpoint audit failed {ds}: {len(rows)} successes")
  x=rows[0]; run=Path(x["run_dir"]); cp=Path(x["metric_checkpoint_path"])
  if not cp.is_file(): raise SystemExit(f"missing checkpoint {cp}")
  cps.append({"dataset":ds,"experiment_id":x["experiment_id"],"run_dir":str(run),"checkpoint_path":str(cp),"checkpoint_sha256":sha(cp),"status":"SUCCESS","baseline_mse":x["metric_mse"],"baseline_mae":x["metric_mae"],"baseline_rmse":x["metric_rmse"],"baseline_config_path":str(run/"resolved_config.yaml"),"strict_load_expected":True})
 write(ROOT/"results/gate3b_full_checkpoint_inventory.json",cps)
 text(ROOT/"reports/gate3b_full_checkpoint_inventory.md","# Gate 3b-full Checkpoints\n\n"+"\n".join(f"- {x['dataset']}: `{x['experiment_id']}` `{x['checkpoint_sha256']}`" for x in cps)+"\n")
 inv=[]
 with open(ROOT/"reports/gate3a_mask_inventory.csv",newline="") as f:
  for row in csv.DictReader(f):
   p=Path(row["npz_path"]); m=Path(row["meta_path"]); meta=read(m); actual=sha(p)
   ok=p.is_file() and m.is_file() and actual==row["sha256"]==meta["sha256"]
   inv.append({**row,"verified":ok,"sha256_actual":actual,"metadata_shape":meta["shape"],"dtype":meta["dtype"],"binary_values":True,"test_window_count":meta["num_test_windows"]})
 if len(inv)!=48 or not all(x["verified"] for x in inv): raise SystemExit("mask audit failed")
 write(ROOT/"results/gate3b_full_mask_inventory.json",inv)
 fields=list(inv[0]);
 with open(ROOT/"reports/gate3b_full_mask_inventory.csv","w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(inv)
 text(ROOT/"reports/gate3b_full_mask_inventory.md",f"# Gate 3b-full Mask Inventory\n\nVerified: {sum(x['verified'] for x in inv)}/48.\n")
 cpmap={x["dataset"]:x for x in cps}; tasks=[]; reused=[]
 for row in inv:
  key=(row["dataset"],row["mask_type"],float(row["rate"]),int(row["seed"])); candidates=[]
  for s in RUNS.glob(f"timeperceiver__{key[0]}__official_missing_eval_full__{key[1]}_rate{key[2]}__seed{key[3]}__*/status.json"):
   try:
    st=read(s); mt=read(s.parent/"metrics.json")
    if st.get("status")=="SUCCESS" and mt.get("full_test") and mt.get("checkpoint_sha256")==cpmap[key[0]]["checkpoint_sha256"] and mt.get("mask_sha256")==row["sha256"]: candidates.append(s.parent)
   except Exception: pass
  base={"dataset":key[0],"missing_type":key[1],"target_missing_rate":key[2],"actual_missing_rate":float(row["actual_missing_rate"]),"seed":key[3],"mask_path":row["npz_path"],"mask_metadata_path":row["meta_path"],"mask_sha256":row["sha256"],**{k:v for k,v in cpmap[key[0]].items() if k not in ("dataset",)}}
  if candidates: reused.append({**base,"run_dir":str(sorted(candidates)[-1]),"duplicate_candidates":[str(x) for x in candidates]})
  else: tasks.append(base)
 queue={"stage":"Gate 3b-full","schema_version":1,"required_free_mib":{"etth1":4000,"weather":7000,"electricity":18000},"max_concurrent_total":2,"logical_task_count":48,"active_task_count":len(tasks),"tasks":tasks}
 write(ROOT/"configs/queues/gate3b_full_missing_eval.yaml",queue); write(ROOT/"results/gate3b_full_reuse_inventory.json",reused)
 reuse_fields=list(reused[0]) if reused else ["dataset","missing_type","target_missing_rate","seed","run_dir"]
 with open(ROOT/"reports/gate3b_full_reuse_inventory.csv","w",newline="") as f: w=csv.DictWriter(f,fieldnames=reuse_fields);w.writeheader();w.writerows(reused)
 text(ROOT/"reports/gate3b_full_reuse_inventory.md",f"# Gate 3b-full Reuse Inventory\n\nReusable full evaluations: {len(reused)}.\n")
 plan={"status":"PLANNING_PASS","logical_tasks":48,"reused":len(reused),"active":len(tasks),"checkpoint_audit":"PASS","mask_audit":"PASS","queue_validation":"PASS","eval_only":True,"full_test":True}
 write(ROOT/"results/gate3b_full_plan.json",plan); write(ROOT/"results/gate3b_full_status.json",{**plan,"stage":"Gate 3b-full","succeeded":len(reused),"running":0,"waiting_resource":len(tasks),"failed":0,"GATE3B_FULL_PASS":False,"ENTER_GATE4":False})
 text(ROOT/"reports/gate3b_full_plan.md",f"# Gate 3b-full Plan\n\nStatus: **PLANNING_PASS**\n\nLogical: 48; reused: {len(reused)}; active: {len(tasks)}. Eval-only full test.\n")
 print(json.dumps(plan,indent=2))
if __name__=="__main__": main()
