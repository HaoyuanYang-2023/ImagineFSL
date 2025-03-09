import torch
import torch.nn as nn

# class GaussPool(nn.Module):
#     def __init__(self, embed_dim, gauss_dim, iterN=3, eps=1e-5):
#         super(GaussPool, self).__init__()
#         self.iterN = iterN
#         self.eps = eps
#         self.dr = nn.Linear(embed_dim, gauss_dim, bias=False)
        
#     # compute covariance matrix
#     def cov(self, x):
#         x = x.transpose(1, 2)
#         bs, C, M = x.size()
#         # x = x.view(bs, C, M)
#         x_mean = torch.mean(x, dim=2, keepdim=True)
#         x = x - x_mean
#         x_cov = x.bmm(x.permute(0, 2, 1)) * 1./(M-1)
#         return x_cov
    
#     # isqrt implementation
#     def sqrtm(self, x, iterN):
#         batchSize = x.data.shape[0]
#         dim = x.data.shape[1]
#         dtype = x.dtype
#         I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
#         normA = (1.0 / 3.0) * x.mul(I3).sum(dim=1).sum(dim=1)
#         A = x.div(normA.view(batchSize, 1, 1).expand_as(x))
#         if iterN < 2:
#             ZY = 0.5 * (I3 - A)
#             YZY = A.bmm(ZY)
#         else:
#             ZY = 0.5 * (I3 - A)
#             Y = A.bmm(ZY)
#             Z = ZY
#             for i in range(1, iterN - 1):
#                 ZY = 0.5 * (I3 - Z.bmm(Y))
#                 Y = Y.bmm(ZY)
#                 Z = ZY.bmm(Z)
#             YZY = 0.5 * Y.bmm(I3 - Z.bmm(Y))
#         y = YZY * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
#         return y
    
#     def triuvec(self, x):
#         batchSize, dim, dim = x.shape
#         r = x.reshape(batchSize, dim * dim)
#         I = torch.ones(dim, dim).triu().reshape(dim * dim)
#         index = I.nonzero(as_tuple = False)
#         y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device).type(x.dtype)
#         y = r[:, index].squeeze()
#         return y

#     def forward(self, x):
#         x = self.dr(x)
#         # mean = x[:,0, :]# x.mean(dim=1) # mean pooling
#         # patch = x[:,1:,:] # patch tokens
        
#         mean = x.mean(dim=1) # mean pooling
#         patch = x
        
#         cov = self.cov(patch) # + self.eps # * torch.eye(128).unsqueeze(0).repeat(x.size(0), 1, 1).to(x.device)
#         # print(cov.abs().max())
#         # import pdb; pdb.set_trace()
        
#         D = mean.size(1)
#         BS = mean.size(0)
        
#         mean_unsq = mean.unsqueeze(2) # (N, D, 1)
#         # import pdb; pdb.set_trace()
#         m1 = cov + mean_unsq @ mean_unsq.transpose(1, 2)
#         m2 = mean_unsq
#         m3 = mean_unsq.transpose(1, 2)
#         m4 = torch.ones(BS, 1, 1).to(mean.device)
#         # print(m1.size(), m2.size(), m3.size(), m4.size())
#         m = torch.cat([torch.cat([m1, m2], dim=2), torch.cat([m3, m4], dim=2)], dim=1)
#         # m: (N, D+1, D+1)
#         gauss_embed = self.sqrtm(m, self.iterN)
#         # get the upper triangular part
#         ge_tri = self.triuvec(gauss_embed).view(BS, -1)
#         return ge_tri
    

# class GaussPool2(nn.Module):
#     def __init__(self, embed_dim=512, gauss_dim=512, iterN=3, eps=1e-5):
#         super(GaussPool2, self).__init__()
#         self.iterN = iterN
#         self.eps = eps
#         # self.dr = nn.Linear(embed_dim, gauss_dim, bias=False)
#         self.dr = nn.Identity()
        
#     # compute covariance matrix
#     # def cov(self, x):
#     #     x = x.transpose(1, 2)
#     #     bs, C, M = x.size()
#     #     # x = x.view(bs, C, M)
#     #     x_mean = torch.mean(x, dim=2, keepdim=True)
#     #     x = x - x_mean
#     #     x_cov = x.bmm(x.permute(0, 2, 1)) * 1./(M-1)
#     #     return x_cov
    
#     # isqrt implementation
#     # def sqrtm(self, x, iterN):
#     #     batchSize = x.data.shape[0]
#     #     dim = x.data.shape[1]
#     #     dtype = x.dtype
#     #     I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
#     #     normA = (1.0 / 3.0) * x.mul(I3).sum(dim=1).sum(dim=1)
#     #     A = x.div(normA.view(batchSize, 1, 1).expand_as(x))
#     #     if iterN < 2:
#     #         ZY = 0.5 * (I3 - A)
#     #         YZY = A.bmm(ZY)
#     #     else:
#     #         ZY = 0.5 * (I3 - A)
#     #         Y = A.bmm(ZY)
#     #         Z = ZY
#     #         for i in range(1, iterN - 1):
#     #             ZY = 0.5 * (I3 - Z.bmm(Y))
#     #             Y = Y.bmm(ZY)
#     #             Z = ZY.bmm(Z)
#     #         YZY = 0.5 * Y.bmm(I3 - Z.bmm(Y))
#     #     y = YZY * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
#     #     return y
    
#     # def triuvec(self, x):
#     #     batchSize, dim, dim = x.shape
#     #     r = x.reshape(batchSize, dim * dim)
#     #     I = torch.ones(dim, dim).triu().reshape(dim * dim)
#     #     index = I.nonzero(as_tuple = False)
#     #     y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device).type(x.dtype)
#     #     y = r[:, index].squeeze()
#     #     return y

#     def forward(self, x):
#         # x = self.dr(x)
#         cls_token = x[:,0, :]
#         patch = x[:,1:,:] # [BS, N_patch, D]
#         mean = torch.mean(patch, dim=1) # [BS, D]
#         # std = torch.sqrt(patch.var(dim=(1)) + self.eps) # [BS, D]
#         # "calculate the second order origin moment"
#         std = torch.sqrt((patch**2).mean(dim=1) + self.eps) # [BS, D]
#         gauss_embed = torch.cat([cls_token, mean, std], dim=-1) # [BS, 3*D]
#         return gauss_embed
    
    
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
        
