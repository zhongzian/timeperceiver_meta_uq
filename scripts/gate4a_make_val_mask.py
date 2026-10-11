#!/usr/bin/env python
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'artifacts/gate4a_val_masks/etth1/random/rate_0.3'; out.mkdir(parents=True,exist_ok=True)
seq,pred,nchan=384,96,7; val_start=12*30*24-seq; val_end=12*30*24+4*30*24; windows=(val_end-val_start)-seq-pred+1
rng=np.random.default_rng(43025); mask=(rng.random((windows,seq,nchan))>=.3).astype(np.uint8); np.savez_compressed(out/'seed_43025.npz',mask=mask)
h=hashlib.sha256((out/'seed_43025.npz').read_bytes()).hexdigest()
meta={'dataset':'etth1','split':'validation','mask_type':'random','target_missing_rate':.3,'actual_rate':float(1-mask.mean()),'seed':43025,'shape':list(mask.shape),'sha256':h,'future_target_used':False,'source':'official validation windows only'}
(out/'seed_43025.metadata.json').write_text(json.dumps(meta,indent=2)+'\n'); print(json.dumps(meta,indent=2))
