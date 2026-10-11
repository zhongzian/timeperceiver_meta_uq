import importlib.util
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('gate4b_official_model',ROOT/'upstream/TimePerceiver/models/TimePerceiver.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);OfficialModel=mod.Model
from .probabilistic_head import ProbabilisticHead, SUPPORTED_VARIANTS

class Model(OfficialModel):
    def __init__(self,configs,variant='official',probabilistic_head_enabled=False):
        super().__init__(configs);self.variant=variant;self.probabilistic_head_enabled=probabilistic_head_enabled
        if variant not in {"official", *SUPPORTED_VARIANTS}:
            raise ValueError(f"invalid variant: {variant}")
        self.probabilistic_head=ProbabilisticHead(self.embed_dim,configs.patch_len,configs.enc_in,variant) if probabilistic_head_enabled else None
    def initialize_probabilistic_location(self):
        if self.probabilistic_head is None:raise RuntimeError('probabilistic head disabled')
        self.probabilistic_head.initialize_location(self.output_projection)
    def forward(self,inputs,x_mark_enc,x_dec,x_mark_dec,indices=None,mask=None):
        if not self.probabilistic_head_enabled:return super().forward(inputs,x_mark_enc,x_dec,x_mark_dec,indices,mask)
        means=inputs.mean(1,keepdim=True).detach();x=inputs-means;stdev=torch.sqrt(torch.var(x,dim=1,keepdim=True,unbiased=False)+1e-5);x=x/stdev
        x=x.transpose(1,2).unfold(2,self.patch_size,self.patch_size);b,c,p,_=x.size();x=self.patch_embedding(x);x=x+self.patch_positional_embedding[:,:,indices[0] if indices else slice(None),:]+self.channel_positional_embedding;x=x.view(b,c*p,self.embed_dim)
        if self.use_latent:
            latent=self.latent_array.expand(b,-1,-1)
            for _ in range(self.num_latent_blocks):
                latent=self.latent_cross_attention(latent,x)
                for block in self.latent_attention_blocks:latent=block(latent,latent)
                x=self.write_cross_attention(x,latent)
        q=self.patch_positional_embedding[:,:,indices[1] if indices else slice(self.past_patch_num,None),:]+self.channel_positional_embedding;q=q.expand(b,-1,-1,-1).contiguous().reshape(b*c,-1,self.embed_dim);x=x.view(b,c,p,self.embed_dim).contiguous().reshape(b*c,-1,self.embed_dim)
        h=self.query_cross_attention(q,x).reshape(b,c,-1,self.embed_dim);loc,scale,df=self.probabilistic_head(h);loc=loc.view(b,c,-1).permute(0,2,1);scale=scale.view(b,c,-1).permute(0,2,1);loc=loc*stdev[:,0,:].unsqueeze(1)+means[:,0,:].unsqueeze(1)
        return {'loc':loc,'scale':scale,'df':df.expand(b,self.pred_len,c) if df is not None else None}

def freeze_backbone(model):
    for n,p in model.named_parameters():p.requires_grad=n.startswith('probabilistic_head.')
    return [n for n,p in model.named_parameters() if p.requires_grad]
