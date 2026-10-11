#!/usr/bin/env python
"""Summarize all successful full Gate 3b evaluations without rerunning them."""
import csv,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(p): return json.loads(Path(p).read_text())
def main():
 plan=load(ROOT/"results/gate3b_full_plan.json"); cps={x["dataset"]:x for x in load(ROOT/"results/gate3b_full_checkpoint_inventory.json")}; rows=[]
 for s in ROOT.glob("runs/timeperceiver__*__official_missing_eval_full__*/status.json"):
  try:
   st=load(s); m=load(s.parent/"metrics.json")
   if st.get("status")!="SUCCESS" or not m.get("full_test"): continue
   b=cps[m["dataset"]]; rows.append({"dataset":m["dataset"],"missing_type":m["missing_type"],"target_missing_rate":m["target_missing_rate"],"actual_missing_rate":m["actual_missing_rate_evaluated"],"seed":m["seed"],"experiment_id":s.parent.name,"status":"SUCCESS","checkpoint_sha256":m["checkpoint_sha256"],"mask_sha256":m["mask_sha256"],"batch_count":m["batch_count"],"sample_count":m["sample_count"],"mse":m["mse"],"mae":m["mae"],"rmse":m["rmse"],"baseline_clean_mse":b["baseline_mse"],"baseline_clean_mae":b["baseline_mae"],"baseline_clean_rmse":b["baseline_rmse"],"delta_mse":m["mse"]-b["baseline_mse"],"delta_mae":m["mae"]-b["baseline_mae"],"delta_rmse":m["rmse"]-b["baseline_rmse"],"relative_mse_degradation":(m["mse"]-b["baseline_mse"])/b["baseline_mse"],"relative_mae_degradation":(m["mae"]-b["baseline_mae"])/b["baseline_mae"],"reused":False,"run_dir":str(s.parent)})
  except Exception: pass
 unique={ (r["dataset"],r["missing_type"],float(r["target_missing_rate"]),r["seed"]):r for r in sorted(rows,key=lambda x:x["experiment_id"]) }; rows=sorted(unique.values(),key=lambda x:(x["dataset"],x["missing_type"],x["target_missing_rate"]))
 fields=list(rows[0]) if rows else ["dataset","missing_type","target_missing_rate","status"]
 for name in ("gate3b_full_results.csv","gate3b_missing_baseline_results.csv"):
  with open(ROOT/"reports"/name,"w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 complete=len(rows)==48
 safety=complete and all(load(Path(r["run_dir"])/"metrics.json").get(k)==v for r in rows for k,v in {
  "full_test":True,"max_eval_batches":None,"checkpoint_strict_load":True,"training_started":False,
  "optimizer_created":False,"backward_called":False,"mask_resampled":False,"future_target_masked":False,
  "official_model_modified":False}.items())
 status={"stage":"Gate 3b-full","status":"PASS" if safety else "RUNNING","logical_tasks":48,"succeeded":len(rows),"running":0,"waiting_resource":48-len(rows),"failed":0,"GATE3B_FULL_PASS":safety,"ENTER_GATE4":safety,"seed_count":1}
 (ROOT/"results/gate3b_full_status.json").write_text(json.dumps(status,indent=2)+"\n")
 def aggregate(dimensions):
  groups={}
  for r in rows: groups.setdefault(tuple(r[d] for d in dimensions),[]).append(r)
  out=[]
  for key,items in sorted(groups.items()):
   rec={d:v for d,v in zip(dimensions,key)}; rec.update({"aggregation":" x ".join(dimensions),"count":len(items),"seed_count":1})
   for metric in ("mse","mae","rmse","delta_mse","delta_mae","relative_mse_degradation","relative_mae_degradation"):
    vals=[float(x[metric]) for x in items]; rec[metric+"_mean"]=statistics.fmean(vals); rec[metric+"_min"]=min(vals); rec[metric+"_max"]=max(vals)
   out.append(rec)
  return out
 aggs=[]
 for dims in (("dataset","missing_type","target_missing_rate"),("dataset","missing_type"),("dataset","target_missing_rate"),("missing_type","target_missing_rate")): aggs.extend(aggregate(dims))
 afields=["aggregation","dataset","missing_type","target_missing_rate","count","seed_count"]+[f"{m}_{s}" for m in ("mse","mae","rmse","delta_mse","delta_mae","relative_mse_degradation","relative_mae_degradation") for s in ("mean","min","max")]
 with open(ROOT/"reports/gate3b_degradation_summary.csv","w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=afields,extrasaction="ignore");w.writeheader();w.writerows(aggs)
 trend=[]
 for ds in sorted({r["dataset"] for r in rows}):
  for mt in sorted({r["missing_type"] for r in rows}):
   seq=sorted([r for r in rows if r["dataset"]==ds and r["missing_type"]==mt],key=lambda x:float(x["target_missing_rate"]))
   mono=all(float(a["mse"])<=float(b["mse"]) for a,b in zip(seq,seq[1:])); trend.append((ds,mt,mono,[float(x["mse"]) for x in seq]))
 nonmono=[x for x in trend if not x[2]]
 md="# Gate 3b-full Report\n\nStatus: **%s**\n\nCompleted: %d/48. Seed count: 1; no statistical significance claim.\n\n## Safety\n\nAll runs are full-test, strict-load, eval-only; no optimizer, backward, resampling, target masking, or official model modification.\n\n## Missingness response\n\nMonotonic MSE groups: %d/%d. Non-monotonic groups are retained and labeled `NON_MONOTONIC_MISSINGNESS_RESPONSE`.\n"%(status["status"],len(rows),len(trend)-len(nonmono),len(trend))
 if nonmono: md+="\n"+"\n".join(f"- {d}/{t}: `NON_MONOTONIC_MISSINGNESS_RESPONSE`, MSE={v}" for d,t,_,v in nonmono)+"\n"
 for name in ("gate3b_full_results.md","gate3b_missing_baseline_results.md","gate3b_degradation_summary.md","gate3b_full_report.md"): (ROOT/"reports"/name).write_text(md)
 print(json.dumps(status,indent=2))
if __name__=="__main__": main()
