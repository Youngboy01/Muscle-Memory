import numpy as np


def overlapping_max_pool2d(
    x: np.ndarray, kernel_size: int = 3, stride: int = 2
) -> np.ndarray:
    """
    Applies overlapping max pooling to a 4D tensor (N, C, H, W).
    Uses ceil mode for output dimensions (allows partial windows at boundaries).

    Args:
        x: Input array of shape (N, C, H, W)
        kernel_size: Size of pooling window (int)
        stride: Stride between pooling windows (int), must be < kernel_size

    Returns:
        A 4D tensor after overlapping pooling with ceil mode.
    """
    # Your code here
    n, c, h, w = x.shape
    out_h = int(np.ceil((h - kernel_size) / stride + 1))
    out_w = int(np.ceil((w - kernel_size) / stride + 1))
    output = np.zeros((n, c, out_h, out_w))
    for i in range(n):
        for j in range(c):
            for k in range(out_h):
                for l in range(out_w):
                    h_start = k * stride
                    w_start = l * stride

                    h_end = min(h_start + kernel_size, h)
                    w_end = min(w_start + kernel_size, w)
                    output[i, j, k, l] = np.max(x[i, j, h_start:h_end, w_start:w_end])
    return output
