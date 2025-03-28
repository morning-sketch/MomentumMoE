import numpy as np
import torch


def cov_to_corr(cov_matrix):
    """
    同时支持numpy数组和PyTorch张量
    """
    is_tensor = torch.is_tensor(cov_matrix)

    if is_tensor:
        assert cov_matrix.dim() == 2  # 矩阵
        device = cov_matrix.device
        diag = torch.sqrt(torch.diag(cov_matrix))
        min_variance = torch.tensor(1e-8, device=device)
    else:
        assert cov_matrix.ndim == 2  # 矩阵
        diag = np.sqrt(np.diag(cov_matrix))
        min_variance = 1e-8

    assert cov_matrix.shape[0] == cov_matrix.shape[1]  # 方阵
    assert (diag > min_variance).all()  # 方差检查

    inv_std = 1.0 / diag

    if is_tensor:
        # PyTorch版本计算
        correlation_matrix = cov_matrix * inv_std.unsqueeze(1) * inv_std.unsqueeze(0)
    else:
        # 原始numpy计算
        correlation_matrix = cov_matrix * inv_std * inv_std[:, np.newaxis]

    return correlation_matrix
