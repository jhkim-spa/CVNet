import torch
import torch.nn as nn


@torch.no_grad()
def pad(cv, size_divisor):
    pad_h = size_divisor - cv.shape[-2] % size_divisor
    pad_w = size_divisor - cv.shape[-1] % size_divisor
    m = nn.ZeroPad2d((pad_w, pad_w, 0, pad_h))
    cv_padded = m(cv)

    return cv_padded


@torch.no_grad()
def resize(cv, size):
    device = cv.device
    w_des, h_des = size
    w_scale = w_des / cv.shape[1]
    h_scale = h_des / cv.shape[0]

    if (w_scale == 1. and h_scale == 1.):
        cv_resized = cv
    else:
        cv_resized = torch.zeros((h_des, w_des, cv.shape[-1]),
                                    dtype=torch.float32, device=device)
        idx_src = torch.nonzero(cv[..., 0], as_tuple=True)
        idx_des = list()
        idx_des.append((h_scale * idx_src[0]).to(torch.long))
        idx_des.append((w_scale * idx_src[1]).to(torch.long))
        cv_resized[idx_des[0], idx_des[1], :] = \
            cv[idx_src[0], idx_src[1], :]

    return cv_resized
