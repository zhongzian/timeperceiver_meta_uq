#!/usr/bin/env python
"""Full-test fixed-mask evaluation using the unmodified official model."""
from __future__ import annotations

import argparse, json, math, os, shutil, sys, traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gate3b_smoke_missing_eval import (build_args, import_official_modules,
    load_checkpoint_strict, parse_simple_yaml, read_json, save_run_scaffolding,
    sha256_file, write_json, write_text)

def now(): return datetime.now().isoformat(timespec="seconds")

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task-file", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--max-eval-batches", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=None,
                   help="Parity/debug only; full queue leaves this unset.")
    a = p.parse_args(); task = read_json(Path(a.task_file)); run = Path(a.run_dir)
    if not run.is_absolute(): run = ROOT / run
    run.mkdir(parents=True, exist_ok=False); (run / "checkpoints").mkdir()
    started = now(); wall = datetime.now()
    cmd = " ".join(sys.argv)
    save_run_scaffolding(run, f"#!/usr/bin/env bash\n# eval-only; no optimizer/backward\n{sys.executable} {cmd}\n")
    write_text(run / "train.log", "Gate 3b-full eval-only. No training, optimizer, or backward.\n")
    status = "FAILED"; error_type = ""; error = ""; metrics = {}
    try:
        mask_path=Path(task["mask_path"]); meta_path=Path(task["mask_metadata_path"])
        checkpoint=Path(task["checkpoint_path"]); config_path=Path(task["baseline_config_path"])
        meta=read_json(meta_path)
        if sha256_file(mask_path) != task["mask_sha256"]: raise RuntimeError("mask hash mismatch")
        if sha256_file(checkpoint) != task["checkpoint_sha256"]: raise RuntimeError("checkpoint hash mismatch")
        z=np.load(mask_path, allow_pickle=False); mask=z["mask"]
        if mask.ndim != 3 or mask.dtype != np.uint8 or not np.isin(mask, [0,1]).all():
            raise RuntimeError("invalid mask array")
        cfg=parse_simple_yaml(config_path)
        eval_batch_size = int(a.eval_batch_size or cfg["batch_size"])
        ev=build_args(cfg, eval_batch_size, a.device)
        provider, model_mod=import_official_modules(); test_data, loader=provider(ev,"test")
        if len(test_data)!=mask.shape[0] or tuple(mask.shape[1:])!=(int(ev.seq_len),int(ev.enc_in)):
            raise RuntimeError(f"mask/data shape mismatch {mask.shape} len={len(test_data)}")
        device=torch.device("cuda:0" if a.device=="cuda" and torch.cuda.is_available() else "cpu")
        model=model_mod.Model(ev).float().to(device); load=load_checkpoint_strict(model,checkpoint); model.eval()
        sse=sae=0.; count=batches=samples=offset=0; rates=[]; pred=int(ev.pred_len); patch=int(ev.patch_len)
        indices=(list(range(int(ev.seq_len)//patch)),list(range(int(ev.seq_len)//patch,(int(ev.seq_len)+pred)//patch)))
        with torch.no_grad():
            for bid,(bx,by,bxm,bym) in enumerate(loader):
                if a.max_eval_batches is not None and bid>=a.max_eval_batches: break
                n=bx.shape[0]; mb=torch.from_numpy(mask[offset:offset+n].astype(np.float32)).to(device)
                bx=bx.float().to(device); by=by.float().to(device); bxm=bxm.float().to(device); bym=bym.float().to(device)
                if tuple(mb.shape)!=tuple(bx.shape): raise RuntimeError("mask batch shape mismatch")
                dec=torch.cat([by[:,:int(ev.label_len),:],torch.zeros_like(by[:,-pred:,:])],1)
                out=model(bx*mb,bxm,dec,bym,indices)[:,-pred:,:]; target=by[:,-pred:,:]
                fdim=-1 if ev.features=="MS" else 0; diff=out[:,:,fdim:]-target[:,:,fdim:]
                sse+=float((diff*diff).sum()); sae+=float(diff.abs().sum()); count+=diff.numel()
                batches+=1; samples+=n; rates.append(1-float(mb.mean())); offset+=n
        full=a.max_eval_batches is None
        mse=sse/count; elapsed=(datetime.now()-wall).total_seconds()
        metrics={**task,"stage":"Gate 3b-full","mode":"official_missing_eval_full","eval_only":True,
          "full_test":full,"max_eval_batches":a.max_eval_batches,"batch_count":batches,"sample_count":samples,
          "actual_missing_rate_evaluated":float(np.mean(rates)),"mse":mse,"mae":sae/count,"rmse":math.sqrt(mse),
          "checkpoint_strict_load":load["strict"],"checkpoint_tensor_count":load["num_tensors"],
          "training_started":False,"optimizer_created":False,"backward_called":False,"mask_resampled":False,
          "masked_input":"batch_x","future_target_masked":False,"official_model_modified":False,
          "device":str(device),"started_at":started,"finished_at":now(),"elapsed_seconds":elapsed}
        shutil.copy2(meta_path,run/"artifacts"/"used_mask_metadata.json")
        write_text(run/"artifacts"/"used_checkpoint.txt",str(checkpoint)+"\n")
        write_text(run/"resolved_config.yaml","\n".join(f"{k}: {json.dumps(v)}" for k,v in sorted({**vars(ev),**task,"full_test":full,"eval_only":True}.items()))+"\n")
        status="SUCCESS"
    except torch.cuda.OutOfMemoryError as e:
        error_type="OOM"; error=repr(e); metrics={**task,"error":error}
    except Exception as e:
        error_type="CODE_ERROR"; error=repr(e); metrics={**task,"error":error,"traceback":traceback.format_exc()}
    write_json(run/"metrics.json",metrics)
    payload={**task,"stage":"Gate 3b-full","experiment_id":run.name,"status":status,"exit_code":0 if status=="SUCCESS" else 2,
      "eval_only":True,"full_test":metrics.get("full_test",False),"training_started":False,"optimizer_created":False,
      "backward_called":False,"mask_resampled":False,"future_target_masked":False,"official_model_modified":False,
      "batch_count":metrics.get("batch_count"),"sample_count":metrics.get("sample_count"),"mse":metrics.get("mse"),
      "mae":metrics.get("mae"),"rmse":metrics.get("rmse"),"started_at":started,"finished_at":now(),
      "error_type":error_type,"error_summary":error}
    write_json(run/"status.json",payload); write_json(run/"resource_usage.json",{"device":metrics.get("device",a.device),"elapsed_seconds":metrics.get("elapsed_seconds"),"eval_only":True})
    write_text(run/"eval.log",json.dumps(payload,indent=2)+"\n")
    return 0 if status=="SUCCESS" else 2
if __name__=="__main__": raise SystemExit(main())
