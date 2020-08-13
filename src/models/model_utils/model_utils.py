import torch

def mask_upper_half_matrix(mat, val=0.0, include_diagonal=False):
    b,h,w = mat.size()
    offset = 0 if include_diagonal else 1
    ind = torch.tru_indices(h, w, offset=offset)
    mat[:,ind[0],ind[1]] = val