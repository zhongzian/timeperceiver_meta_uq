import sys, importlib.util
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
_spec=importlib.util.spec_from_file_location('official_timeperceiver_model',ROOT/'upstream'/'TimePerceiver'/'models'/'TimePerceiver.py')
_mod=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
OfficialModel=_mod.Model
from .missingness_adapter import MissingnessEncoder, ResidualBottleneckAdapter

class Model(OfficialModel):
    def __init__(self, configs, variant='missingness_aware_adapter'):
        super().__init__(configs); self.variant=variant
        if variant not in {'official','static_adapter','missingness_aware_adapter'}: raise ValueError('invalid Gate 4a variant')
        self.adapter=ResidualBottleneckAdapter(self.embed_dim,max(8,self.embed_dim//4),32 if variant=='missingness_aware_adapter' else None) if variant!='official' else None
        self.missingness_encoder=MissingnessEncoder() if variant=='missingness_aware_adapter' else None
    def forward(self, inputs,x_mark_enc,x_dec,x_mark_dec,indices=None,mask=None):
        means=inputs.mean(1,keepdim=True).detach(); inputs=inputs-means; stdev=torch.sqrt(torch.var(inputs,dim=1,keepdim=True,unbiased=False)+1e-5); inputs=inputs/stdev
        inputs=inputs.transpose(1,2).unfold(2,self.patch_size,self.patch_size); b,c,p,ps=inputs.size(); inputs=self.patch_embedding(inputs)
        inputs=inputs+self.patch_positional_embedding[:,:,indices[0] if indices else slice(None),:]+self.channel_positional_embedding; inputs=inputs.view(b,c*p,self.embed_dim)
        if self.use_latent:
            latent=self.latent_array.expand(b,-1,-1)
            for _ in range(self.num_latent_blocks):
                latent=self.latent_cross_attention(latent,inputs)
                for block in self.latent_attention_blocks: latent=block(latent,latent)
                inputs=self.write_cross_attention(inputs,latent)
        q=self.patch_positional_embedding[:,:,indices[1] if indices else slice(self.past_patch_num,None),:]+self.channel_positional_embedding
        q=q.expand(b,-1,-1,-1).contiguous().reshape(b*c,-1,self.embed_dim); inputs=inputs.view(b,c,p,self.embed_dim).contiguous().reshape(b*c,-1,self.embed_dim)
        outputs=self.query_cross_attention(q,inputs).reshape(b,c,-1,self.embed_dim)
        if self.adapter is not None: outputs=self.adapter(outputs,self.missingness_encoder(mask) if self.missingness_encoder is not None else None)
        outputs=self.output_projection(outputs).view(b,c,-1).contiguous().permute(0,2,1)
        return outputs*stdev[:,0,:].unsqueeze(1).repeat(1,self.pred_len,1)+means[:,0,:].unsqueeze(1).repeat(1,self.pred_len,1)

def freeze_backbone(model):
    for n,p in model.named_parameters(): p.requires_grad=n.startswith('adapter.') or n.startswith('missingness_encoder.')
    return [n for n,p in model.named_parameters() if p.requires_grad]
