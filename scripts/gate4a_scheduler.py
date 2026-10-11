#!/usr/bin/env python
"""Queue-specific resumable scheduler for the two Gate 4a formal tasks."""
import argparse,json,os,subprocess,sys,time
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PY='/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python';STATE=ROOT/'runtime/gate4a_minimal_adapter_scheduler_state.json'
def load(p): return json.loads(Path(p).read_text())
def save(x): STATE.parent.mkdir(exist_ok=True);STATE.write_text(json.dumps(x,indent=2)+'\n')
def gpu_free():
    try:
        out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        return sorted([(int(r.split(',')[1]),int(r.split(',')[0])) for r in out.splitlines()],reverse=True)
    except Exception:return []
def success_for(t):
    pat=f"timeperceiver__etth1__gate4a__{t['variant']}__random_rate0.3__seed2025__*/status.json"
    for p in sorted((ROOT/'runs').glob(pat)):
        try:
            if load(p).get('status')=='SUCCESS': return p.parent
        except Exception:pass
    return None
def main():
    p=argparse.ArgumentParser();p.add_argument('queue');a=p.parse_args();queue=Path(a.queue);queue=queue if queue.is_absolute() else ROOT/queue;q=load(queue);tasks=q['tasks'];state={'queue':str(queue),'scheduler_pid':os.getpid(),'status':'RUNNING','tasks':{t['variant']:{'status':'PENDING'} for t in tasks},'updated_at':datetime.now().isoformat()};save(state)
    for index,t in enumerate(tasks):
        old=success_for(t)
        if old: state['tasks'][t['variant']]={'status':'SUCCEEDED','run_dir':str(old),'reused':True};save(state);continue
        retry=0
        while True:
            cards=[x for x in gpu_free() if x[0]>=int(t['required_free_mib'])]
            if not cards:
                state['tasks'][t['variant']]={'status':'WAITING_RESOURCE','required_free_mib':t['required_free_mib']};state['status']='WAITING_RESOURCE';state['updated_at']=datetime.now().isoformat();save(state);print(f"WAITING_RESOURCE variant={t['variant']} required={t['required_free_mib']}",flush=True);time.sleep(60);continue
            gpu=cards[0][1];ts=datetime.now().strftime('%Y%m%d_%H%M%S');suffix=f'__retry{retry}' if retry else '';rid=f"timeperceiver__etth1__gate4a__{t['variant']}__random_rate0.3__seed2025__{ts}{suffix}";tf=ROOT/'runtime'/f"gate4a_task_{index}.json";tf.write_text(json.dumps(t,indent=2)+'\n');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);cmd=[PY,'-u',str(ROOT/'scripts/gate4a_formal_runner.py'),'--task-file',str(tf),'--run-dir',str(ROOT/'runs'/rid)];proc=subprocess.Popen(cmd,env=env,cwd=str(ROOT));state['status']='RUNNING';state['tasks'][t['variant']]={'status':'RUNNING','experiment_id':rid,'run_pid':proc.pid,'gpu':gpu,'epoch':None,'best_validation_mse':None};save(state);print(f'START {rid} pid={proc.pid} gpu={gpu}',flush=True);rc=proc.wait()
            if rc==0: state['tasks'][t['variant']].update({'status':'SUCCEEDED','run_pid':None});save(state);print(f'END {rid} rc=0',flush=True);break
            run=ROOT/'runs'/rid;err='';
            try: err=load(run/'metrics.json').get('error','')
            except Exception: pass
            if 'out of memory' in str(err).lower() and retry<3:
                retry+=1;state['tasks'][t['variant']]={'status':'WAITING_RESOURCE','oom_retry_count':retry,'last_error':err};save(state);time.sleep(120);continue
            state['tasks'][t['variant']].update({'status':'FAILED','exit_code':rc,'last_error':err});state['status']='FAILED';save(state);return rc
    state['status']='COMPLETE';state['updated_at']=datetime.now().isoformat();save(state);print('QUEUE_COMPLETE',flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
