# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
import math

class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class MEMAXLoss(nn.Module):
    
    def __init__(self,student_temp):
        super().__init__()
        self.student_temp = student_temp

    def forward(self, student_output):
        """
        Mean Entropy Maximization Loss of student output
        """
        # import pdb; pdb.set_trace()
        total_loss = 0
        probs = F.softmax(student_output / self.student_temp, dim=-1)
        avg_probs = AllReduce.apply(torch.mean(probs, dim=0))
        rloss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
        return rloss