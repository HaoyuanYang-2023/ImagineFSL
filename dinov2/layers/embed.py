import torch
import torch.nn as nn

    
class HoMPool(nn.Module):
    def __init__(self):
        super(HoMPool, self).__init__()
        self.eps = 1e-5
        
    def forward(self, x):
        cls_token = x[:, 0, :]
        patch = x[:, 1:, :]  # [BS, N_patch, D]
        mean = torch.mean(patch, dim=1)  # [BS, D]
        
        # [BS, D], unbiased=False means the variance is calculated by 1/N instead of 1/(N-1)
        variance = torch.var(patch, dim=1, unbiased=False) 
        std = torch.sqrt(variance + self.eps)  # [BS, D]
        # Calculate third central moment
        centered_patch = patch - mean.unsqueeze(1)  # [BS, N_patch, D]
        third_central_moment = torch.mean(centered_patch ** 3, dim=1)  # [BS, D]
                
        # This approach ensures the resulting cube root retains the direction (positive or negative) of the o
        # riginal third central moment, which is important when dealing with higher-order statistics
        # that may involve negative values.
        third_central_moment = torch.sign(third_central_moment) * (torch.abs(third_central_moment)+ self.eps) ** (1/3)  # [BS, D]
        gauss_embed = torch.cat([cls_token, mean, std, third_central_moment], dim=-1)  # [BS, 4*D]
        return gauss_embed      
        
