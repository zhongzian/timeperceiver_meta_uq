import torch
from torch import nn

class MissingnessEncoder(nn.Module):
    def __init__(self, condition_dim=7, hidden_dim=32):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(condition_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,hidden_dim))
    def forward(self, mask):
        m=1.0-mask.float(); c=m.mean(1); t=m.mean(2)
        x=torch.stack([m.mean((1,2)),c.mean(1),c.std(1,unbiased=False),c.amax(1),t.mean(1),t.std(1,unbiased=False),t.amax(1)],1)
        return self.mlp(x)

class ResidualBottleneckAdapter(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim, condition_dim=None):
        super().__init__()
        self.norm=nn.LayerNorm(hidden_dim); self.down=nn.Linear(hidden_dim,bottleneck_dim); self.up=nn.Linear(bottleneck_dim,hidden_dim)
        self.condition=nn.Linear(condition_dim,2*bottleneck_dim) if condition_dim else None
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
        if self.condition: nn.init.zeros_(self.condition.weight); nn.init.zeros_(self.condition.bias)
    def forward(self,h,z=None):
        u=torch.nn.functional.gelu(self.down(self.norm(h)))
        if self.condition is not None:
            g,b=self.condition(z).unsqueeze(1).unsqueeze(1).chunk(2,-1); u=(1+g)*u+b
        return h+self.up(u)
