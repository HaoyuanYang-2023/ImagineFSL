# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random



def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    """
    Samples list is a list of tuples, each tuple contains a dictionary of samples and a dictionary of targets.
    Each tuple:
    (samples, targets), e.g.,: 
    {'global_crops': [global_crops], 'local_crops': [local_crops], ...} [label_global_1, label_global_2, ...]
    """
    
    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])
    
    collated_global_crops = torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])
    """
    Eual to: Loop over global crops in outer loop, and loop over samples in inner loop
    # Initialize an empty list to hold the tensors
    tensors_list = []
    # Loop over the range of n_global_crops
    for i in range(n_global_crops):
        # Loop over the samples_list
        for s in samples_list:
            # Append the tensor to the list
            tensors_list.append(s[0]["global_crops"][i])
    # Stack the tensors
    stacked_tensors = torch.stack(tensors_list)
    """
    # collated_global_labels = torch.stack([sample[1][i] for i in range(n_global_crops) for sample in samples_list])
    
    if n_local_crops != 0:
        collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])
        collated_local_labels = torch.stack([sample[1][i] for i in range(n_local_crops) for sample in samples_list])
    else:
        collated_local_crops = None
        collated_local_labels = None
    
    B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    # print(f"n_samples_masked: {n_samples_masked}")
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    # prob = mask_ratio_tuple[0]
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))
    
    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()


    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype) if collated_local_crops is not None else None,
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }




