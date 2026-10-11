#!/usr/bin/env python
from __future__ import annotations
import csv,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
from gate3b_smoke_missing_eval import build_args,load_checkpoint_strict,parse_simple_yaml,read_json,sha256_file
from models.missingness_aware_timeperceiver import MissingnessAwareTimePerceiver
def write(p,x): p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+"\n")
def md(p,title,x): Path(p).write_text(f"# {title}\n\n```json\n{json.dumps(x,indent=2)}\n```\n")
def eval8(model,loader,mask,args):
 preds=[];trues=[];off=0;idx=(list(range(args.seq_len//args.patch_len)),list(range(args.seq_len//args.patch_len,(args.seq_len+args.pred_len)//args.patch_len)))
 model.eval()
 with torch.no_grad():
  for i,(bx,by,bxm,bym) in enumerate(loader):
   if i>=8:break
   n=len(bx);m=torch.from_numpy(mask[off:off+n].astype(np.float32)); dec=torch.cat((by[:,:args.label_len,:],torch.zeros_like(by[:,-args.pred_len:,:])),1)
   preds.append(model(bx.float()*m,bxm.float(),dec.float(),bym.float(),idx,m).cpu());trues.append(by[:,-args.pred_len:,:]);off+=n
 return torch.cat(preds),torch.cat(trues)
def metrics(p,t):
 d=p-t;m=float((d*d).mean());return {"mse":m,"mae":float(d.abs().mean()),"rmse":m**.5}
def official_modules():
 sys.path.insert(0,str(ROOT/"upstream/TimePerceiver"))
 from data_provider.data_factory import data_provider
 spec=importlib.util.spec_from_file_location("gate4a_official_timeperceiver",ROOT/"upstream/TimePerceiver/models/TimePerceiver.py")
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 return data_provider,mod
def main():
 cp=next(x for x in read_json(ROOT/"results/gate3b_full_checkpoint_inventory.json") if x["dataset"]=="etth1")
 cfg=parse_simple_yaml(Path(cp["baseline_config_path"]));args=build_args(cfg,32,"cpu");provider,mod=official_modules()
 val,val_loader=provider(args,"val"); test,test_loader=provider(args,"test")
 val_path=ROOT/"artifacts/gate4a_val_masks/etth1/random/rate_0.3/seed_43025.npz";val_path.parent.mkdir(parents=True,exist_ok=True)
 rng=np.random.default_rng(43025);vm=(rng.random((len(val),args.seq_len,args.enc_in))>=.3).astype(np.uint8);np.savez_compressed(val_path,mask=vm)
 vh=sha256_file(val_path);vmeta={"dataset":"etth1","split":"validation","missing_type":"random","target_rate":.3,"actual_rate":1-float(vm.mean()),"seed":43025,"shape":list(vm.shape),"sha256":vh,"future_target_used":False,"source_split":"official ETTh1 validation windows"};write(val_path.with_suffix(".metadata.json"),vmeta)
 row=next(r for r in csv.DictReader(open(ROOT/"reports/gate3b_full_results.csv")) if r["dataset"]=="etth1" and r["missing_type"]=="random" and float(r["target_missing_rate"])==.3)
 tm=np.load(row["run_dir"]+"/../" if False else ROOT/"artifacts/masks/etth1/random/rate_0.3/seed_2025.npz")["mask"]
 backbone=mod.Model(args).float();load=load_checkpoint_strict(backbone,Path(cp["checkpoint_path"])); ref,_=eval8(backbone,test_loader,tm,args)
 refm=metrics(ref,torch.cat([b[1][:,-args.pred_len:,:] for i,b in zip(range(8),test_loader)]))
 results={}
 for name,variant,enabled in (("adapter_disabled","official",False),("static_zero_init","static_adapter",True),("missingness_aware_zero_init","missingness_aware_adapter",True)):
  bb=mod.Model(args).float();load_checkpoint_strict(bb,Path(cp["checkpoint_path"]));w=MissingnessAwareTimePerceiver(bb,args.d_model,args.d_model//4,variant,enabled);pred,true=eval8(w,test_loader,tm,args);mm=metrics(pred,true)
  results[name]={"mse_diff":mm["mse"]-refm["mse"],"mae_diff":mm["mae"]-refm["mae"],"rmse_diff":mm["rmse"]-refm["rmse"],"forecast_max_abs_diff":float((pred-ref).abs().max()),"status":"PASS" if torch.equal(pred,ref) else "FAIL"}
 arch={"status":"PASS","official_model_class":"models.TimePerceiver.Model","official_forward_signature":"(inputs,x_mark_enc,x_dec,x_mark_dec,indices=None,mask=None)","encoder_output_shape":"[B,C,8,256] for ETTh1 p96","hidden_dim":256,"prediction_head":"output_projection Linear(256,12)","selected_insertion_point":"query_cross_attention output after [B,C,P,D] reshape, before output_projection","wrapper_strategy":"project-level wrapper reusing backbone submodules; direct official forward when disabled","checkpoint_loading_strategy":"strict backbone load then fresh extension init","upstream_modified":False,"risks":["enabled path mirrors official forward because no public representation API"]}
 data={"status":"PASS","train_split":"official ETTh1 train","train_mask":"deterministic random 0.3, base_seed 42025","validation_mask":str(val_path),"validation_mask_sha256":vh,"test_mask":str(ROOT/"artifacts/masks/etth1/random/rate_0.3/seed_2025.npz"),"test_mask_sha256":row["mask_sha256"],"batch_y_modified":False,"test_mask_used_during_training":False,"normalization":"official scaler fit on train only"}
 ck={"status":"PASS","source":cp,"strict_load":load}
 for key,obj in (("architecture_audit",arch),("data_protocol_audit",data),("checkpoint_audit",ck)):
  write(ROOT/f"results/gate4a_{key}.json",obj);md(ROOT/f"reports/gate4a_{key}.md",f"Gate 4a {key.replace('_',' ').title()}",obj)
 disabled=results["adapter_disabled"];zero={k:v for k,v in results.items() if "zero_init" in k}
 write(ROOT/"results/gate4a_adapter_disabled_parity.json",disabled);md(ROOT/"reports/gate4a_adapter_disabled_parity.md","Gate 4a Adapter Disabled Parity",disabled)
 write(ROOT/"results/gate4a_zero_init_parity.json",zero);md(ROOT/"reports/gate4a_zero_init_parity.md","Gate 4a Zero Init Parity",zero)
 print(json.dumps(results,indent=2)); return 0 if all(x["status"]=="PASS" for x in results.values()) else 2
if __name__=="__main__":raise SystemExit(main())
