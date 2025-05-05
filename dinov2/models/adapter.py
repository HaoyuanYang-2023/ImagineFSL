import torch
import torch.nn as nn
from torch import Tensor
from dinov2.layers import Mlp, MemEffAttentionDr, NestedTensorBlock as Block
from functools import partial
    
class AdapterDr(nn.Module):
    def __init__(self, embed_dim=512, qkv_dim=384,
                 num_heads=6, mlp_ratio=2., 
                 qkv_bias=True, ffn_bias=True, proj_bias=True,
                 drop_path_rate=0., depth=4,
                 act_layer=nn.GELU, 
                 ffn_layer=Mlp, 
                 norm_layer=nn.LayerNorm,
                 drop_path_uniform=True,
                 ):
        super().__init__()
        
        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
            
        block_fn = partial(Block, attn_class=MemEffAttentionDr)
        

        blocks_list = [
            block_fn(
                dim=embed_dim,
                qkv_dim=qkv_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=0,
            )
            for i in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks_list)


    def forward_features_list(self, x_list):
        x = x_list
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x in all_x:
            output.append(
                {
                    "x_norm_clstoken": x[:, 0],
                    "x_norm_patchtokens": x[:,1:],
                }
            )
            
        return output
    
    def forward_features(self, x): 
        if isinstance(x, list):
            return self.forward_features_list(x)

        for blk in self.blocks:
            x = blk(x)
                
        return {
            "x_norm_clstoken": x[:, 0],
            "x_norm_patchtokens": x[:,1:],
        }
    
    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        return ret
    
  

