#!/usr/bin/env python
"""Gate 4a adapter-only training runner (smoke or formal)."""
import argparse, json, math, os, socket, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from gate3b_smoke_missing_eval import parse_simple_yaml,build_args,import_official_modules,load_checkpoint_strict,sha256_file
from models.missingness_aware_timeperceiver import Model,freeze_backbone

def deterministic_mask(shape,seed):
    return (torch.rand(shape,generator=torch.Generator().manual_seed(seed))>=.3).float()
def dump(path,obj): path.write_text(json.dumps(obj,indent=2)+'\n')
def cmdout(cmd):
    try:return subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT).strip()
    except Exception as e:return repr(e)
def extension_state(model): return {n:v.detach().cpu() for n,v in model.state_dict().items() if n.startswith(('adapter.','missingness_encoder.'))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['static_adapter','missingness_aware_adapter'],required=True);p.add_argument('--run-dir',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--device',choices=['cuda'],default='cuda');a=p.parse_args()
    run=Path(a.run_dir);run.mkdir(parents=True,exist_ok=False);(run/'checkpoints').mkdir(); started=time.time()
    command=' '.join(sys.argv);(run/'command.sh').write_text('#!/usr/bin/env bash\n'+sys.executable+' '+command+'\n');(run/'train.log').write_text('Gate 4a adapter-only GPU training smoke.\n')
    cfgpath=ROOT/'runs/timeperceiver__etth1__official_baseline__seed2025__20260701_203054/resolved_config.yaml';cfg=parse_simple_yaml(cfgpath);args=build_args(cfg,int(cfg['batch_size']),'cuda')
    resolved={**cfg,'variant':a.variant,'smoke_only':a.smoke,'max_epochs':2 if a.smoke else 20,'max_train_batches':32 if a.smoke else None,'max_val_batches':8 if a.smoke else None,'max_test_batches':8 if a.smoke else None,'freeze_backbone':True,'learning_rate_adapter':1e-3,'weight_decay_adapter':1e-4,'gradient_clip':1.0,'train_mask_seed':42025,'validation_mask_seed':43025,'test_mask_seed':2025}
    (run/'resolved_config.yaml').write_text('\n'.join(f'{k}: {json.dumps(v)}' for k,v in sorted(resolved.items()))+'\n')
    (run/'env.txt').write_text(f'hostname: {socket.gethostname()}\npython: {sys.version.split()[0]}\ntorch: {torch.__version__}\ncuda_runtime: {torch.version.cuda}\nCUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES","")}\ngpu_before: {cmdout(["nvidia-smi","--query-gpu=index,uuid,name,memory.total,memory.used,memory.free","--format=csv,noheader"])}\n')
    (run/'git_state.txt').write_text(cmdout(['git','-C',str(ROOT/'upstream/TimePerceiver'),'rev-parse','HEAD'])+'\n'+cmdout(['git','-C',str(ROOT/'upstream/TimePerceiver'),'status','--short'])+'\n');(run/'code.diff').write_text(cmdout(['git','-C',str(ROOT/'upstream/TimePerceiver'),'diff','--binary'])+'\n')
    status='FAILED';metrics={}; error=''
    try:
        if not torch.cuda.is_available(): raise RuntimeError('GPU_REQUIRED')
        torch.manual_seed(2025);np.random.seed(2025);dev=torch.device('cuda:0');torch.cuda.set_device(0);torch.cuda.init();torch.cuda.reset_peak_memory_stats(dev)
        provider,official_mod=import_official_modules();train_data,train_loader=provider(args,'train');val_data,_=provider(args,'val');test_data,test_loader=provider(args,'test')
        val_loader=torch.utils.data.DataLoader(val_data,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,drop_last=False,pin_memory=True)
        cp=next((ROOT/'runs/timeperceiver__etth1__official_baseline__seed2025__20260701_203054/checkpoints').rglob('checkpoint.pth'));raw=torch.load(cp,map_location='cpu')
        official=official_mod.Model(args).float();strict_info=load_checkpoint_strict(official,cp);del official
        model=Model(args,a.variant).float();missing,unexpected=model.load_state_dict(raw,strict=False);expected=all(n.startswith(('adapter.','missingness_encoder.')) for n in missing) and not unexpected
        if not expected: raise RuntimeError(f'extension load mismatch missing={missing} unexpected={unexpected}')
        trainable_names=freeze_backbone(model);model.to(dev);trainable=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(trainable,lr=1e-3,weight_decay=1e-4)
        optimizer_backbone=sum(1 for g in opt.param_groups for p0 in g['params'] if not p0.requires_grad)
        initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad};crit=torch.nn.MSELoss();maxep=2 if a.smoke else 20;maxtrain=32 if a.smoke else None;maxval=8 if a.smoke else None;maxtest=8 if a.smoke else None
        best=float('inf');bestep=0;finite_loss=True;finite_val=True;all_grad_finite=True;adapter_nonzero=False;condition_nonzero=(a.variant=='static_adapter');backbone_grad_count=0;train_batches=0;batch_y_unchanged=True;bad_epochs=0;early_stopped=False;epochs_completed=0
        idx=(list(range(args.seq_len//args.patch_len)),list(range(args.seq_len//args.patch_len,(args.seq_len+args.pred_len)//args.patch_len)))
        def forward(bx,by,bxm,bym,mb):
            dec=torch.cat([by[:,:args.label_len,:],torch.zeros_like(by[:,-args.pred_len:,:])],1);target=by[:,-args.pred_len:,:];return model(bx,bxm,dec,bym,idx,mask=mb)[:,-args.pred_len:,:],target
        for ep in range(maxep):
            epochs_completed=ep+1
            model.train()
            for bi,(bx,by,bxm,bym) in enumerate(train_loader):
                if maxtrain is not None and bi>=maxtrain:break
                bx=bx.float().to(dev);by=by.float().to(dev);bxm=bxm.float().to(dev);bym=bym.float().to(dev);before=by.clone();mb=deterministic_mask(bx.shape,42025+ep*100000+bi).to(dev)
                opt.zero_grad();out,target=forward(bx,by,bxm,bym,mb);loss=crit(out,target);finite_loss&=bool(torch.isfinite(loss));loss.backward();batch_y_unchanged&=bool(torch.equal(by,before));train_batches+=1
                for n,p0 in model.named_parameters():
                    if p0.requires_grad and p0.grad is not None:
                        all_grad_finite&=bool(torch.isfinite(p0.grad).all());nz=bool(torch.count_nonzero(p0.grad))
                        if n.startswith('adapter.'):adapter_nonzero|=nz
                        if n.startswith('missingness_encoder.') or n.startswith('adapter.condition.'):condition_nonzero|=nz
                    elif not p0.requires_grad and p0.grad is not None:backbone_grad_count+=1
                clip_grad_norm_(trainable,1.0);opt.step()
            model.eval();vals=[];vm=np.load(ROOT/'artifacts/gate4a_val_masks/etth1/random/rate_0.3/seed_43025.npz')['mask']
            with torch.no_grad():
                offset=0
                for bi,(bx,by,bxm,bym) in enumerate(val_loader):
                    if maxval is not None and bi>=maxval:break
                    bx=bx.float().to(dev);by=by.float().to(dev);bxm=bxm.float().to(dev);bym=bym.float().to(dev);mb=torch.from_numpy(vm[offset:offset+len(bx)]).float().to(dev);offset+=len(bx);out,target=forward(bx,by,bxm,bym,mb);vals.append(float(crit(out,target)))
            val=float(np.mean(vals));finite_val&=bool(np.isfinite(val))
            if val<best:
                best=val;bestep=ep+1;bad_epochs=0;torch.save({'adapter_state_dict':extension_state(model),'epoch':bestep,'best_validation_mse':best,'variant':a.variant,'source_checkpoint_sha256':sha256_file(cp)},run/'checkpoints/best_adapter.pt')
            else:
                bad_epochs+=1
                if bad_epochs>=3:
                    early_stopped=True
                    break
        torch.save({'adapter_state_dict':extension_state(model),'epoch':maxep,'variant':a.variant,'source_checkpoint_sha256':sha256_file(cp)},run/'checkpoints/last_adapter.pt')
        updated=[n for n,p0 in model.named_parameters() if p0.requires_grad and not torch.equal(initial[n],p0.detach().cpu())]
        bestobj=torch.load(run/'checkpoints/best_adapter.pt',map_location='cpu');reload_model=Model(args,a.variant).float();miss2,unexp2=reload_model.load_state_dict(raw,strict=False);expected_extension=set(extension_state(reload_model));reload_result=reload_model.load_state_dict(bestobj['adapter_state_dict'],strict=False);reload_ok=set(bestobj['adapter_state_dict'])==expected_extension and not reload_result.unexpected_keys and all(n not in expected_extension for n in reload_result.missing_keys)
        reload_model.to(dev).eval();tm=np.load(ROOT/'artifacts/masks/etth1/random/rate_0.3/seed_2025.npz')['mask'];sse=sae=0.;count=test_batches=offset=0
        with torch.no_grad():
            for bi,(bx,by,bxm,bym) in enumerate(test_loader):
                if maxtest is not None and bi>=maxtest:break
                bx=bx.float().to(dev);by=by.float().to(dev);bxm=bxm.float().to(dev);bym=bym.float().to(dev);mb=torch.from_numpy(tm[offset:offset+len(bx)]).float().to(dev);offset+=len(bx);dec=torch.cat([by[:,:args.label_len,:],torch.zeros_like(by[:,-args.pred_len:,:])],1);pred=reload_model(bx,bxm,dec,bym,idx,mask=mb);diff=pred-by[:,-args.pred_len:,:];sse+=float((diff*diff).sum());sae+=float(diff.abs().sum());count+=diff.numel();test_batches+=1
        mse=sse/count;mae=sae/count;finite_test=all(np.isfinite(x) for x in (mse,mae,math.sqrt(mse)))
        pass_all=finite_loss and finite_val and finite_test and all_grad_finite and adapter_nonzero and condition_nonzero and backbone_grad_count==0 and optimizer_backbone==0 and bool(updated) and batch_y_unchanged and reload_ok
        metrics={'variant':a.variant,'dataset':'etth1','seed':2025,'smoke_only':a.smoke,'train_batches':train_batches,'val_batches':min(len(val_loader),maxval or len(val_loader)),'test_batches':test_batches,'finite_loss':finite_loss,'finite_validation_loss':finite_val,'finite_test_metrics':finite_test,'all_new_parameter_gradients_finite':all_grad_finite,'adapter_nonzero_gradient':adapter_nonzero,'condition_nonzero_gradient':condition_nonzero,'backbone_non_null_gradients':backbone_grad_count,'optimizer_backbone_params':optimizer_backbone,'updated_parameter_names':updated,'batch_y_modified':not batch_y_unchanged,'test_mask_used_during_training':False,'no_leakage':batch_y_unchanged,'checkpoint_save':True,'checkpoint_strict_reload':reload_ok,'evaluation_after_reload':finite_test,'max_epochs':maxep,'patience':3,'epochs_completed':epochs_completed,'best_epoch':bestep,'best_validation_mse':best,'early_stopped':early_stopped,'bad_epoch_count_at_stop':bad_epochs,'early_stopping_metric':'validation_mse','test_used_for_model_selection':False,'test_mse':mse,'test_mae':mae,'test_rmse':math.sqrt(mse),'trainable_parameter_count':sum(p.numel() for p in model.parameters() if p.requires_grad),'total_parameter_count':sum(p.numel() for p in model.parameters()),'trainable_parameter_names':trainable_names,'source_checkpoint_sha256':sha256_file(cp),'test_mask_sha256':sha256_file(ROOT/'artifacts/masks/etth1/random/rate_0.3/seed_2025.npz'),'validation_mask_sha256':sha256_file(ROOT/'artifacts/gate4a_val_masks/etth1/random/rate_0.3/seed_43025.npz'),'peak_gpu_memory_mib':torch.cuda.max_memory_allocated(dev)/1048576,'elapsed_seconds':time.time()-started}
        status='SUCCESS' if pass_all else 'FAILED'
    except Exception as exc:
        error=repr(exc);metrics={'variant':a.variant,'smoke_only':a.smoke,'error':error,'traceback':traceback.format_exc(),'elapsed_seconds':time.time()-started}
    status_payload={'status':status,'stage':'gate4a','experiment_id':run.name,'variant':a.variant,'dataset':'etth1','seed':2025,'smoke_only':a.smoke,'source_checkpoint':str(ROOT/'runs/timeperceiver__etth1__official_baseline__seed2025__20260701_203054/checkpoints/ETTh1_384_96_TimePerceiver_sl384_pl96_dm256_nl32_ld64_ul1_gf1_nh4_df512_pl12_qs1_sr1.0/checkpoint.pth'),'source_checkpoint_sha256':metrics.get('source_checkpoint_sha256'),'validation_mask':str(ROOT/'artifacts/gate4a_val_masks/etth1/random/rate_0.3/seed_43025.npz'),'validation_mask_sha256':metrics.get('validation_mask_sha256'),'test_mask':str(ROOT/'artifacts/masks/etth1/random/rate_0.3/seed_2025.npz'),'test_mask_sha256':metrics.get('test_mask_sha256'),'freeze_backbone':True,'trainable_parameter_count':metrics.get('trainable_parameter_count'),'optimizer_backbone_parameter_count':metrics.get('optimizer_backbone_params'),'backbone_non_null_gradient_count':metrics.get('backbone_non_null_gradients'),'max_epochs':metrics.get('max_epochs'),'patience':metrics.get('patience'),'epochs_completed':metrics.get('epochs_completed'),'best_epoch':metrics.get('best_epoch'),'best_validation_mse':metrics.get('best_validation_mse'),'early_stopped':metrics.get('early_stopped'),'early_stopping_metric':metrics.get('early_stopping_metric'),'test_used_for_model_selection':metrics.get('test_used_for_model_selection'),'test_mse':metrics.get('test_mse'),'test_mae':metrics.get('test_mae'),'test_rmse':metrics.get('test_rmse'),'batch_y_modified':metrics.get('batch_y_modified'),'test_mask_used_during_training':metrics.get('test_mask_used_during_training'),'training_started':bool(metrics and not metrics.get('error')),'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'error_type':'CODE_ERROR' if error else '','error_summary':error}
    dump(run/'metrics.json',metrics);dump(run/'status.json',status_payload);dump(run/'resource_usage.json',{'peak_gpu_memory_mib':metrics.get('peak_gpu_memory_mib'),'elapsed_seconds':metrics.get('elapsed_seconds'),'device':'cuda'});(run/'train.log').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps({'status':status,**metrics},indent=2));return 0 if status=='SUCCESS' else 2
if __name__=='__main__':raise SystemExit(main())
