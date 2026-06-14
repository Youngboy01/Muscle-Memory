import numpy as np


def conv3d_forward_pass(
    input_volume: np.ndarray,
    kernel: np.ndarray,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Perform 3D convolution forward pass.

    Slide a 3D kernel over input volume, computing dot products.

    Args:
            input_volume: Shape (C, D, H, W)
              C = channels, D = depth/time, H = height, W = width
            kernel: Shape (C, kD, kH, kW)
              Must match input channels
            stride: (stride_d, stride_h, stride_w)
              Step size in each dimension
            padding: (pad_d, pad_h, pad_w)
              Zero-padding in each dimension

    Returns:
            Output volume: Shape (1, D_out, H_out, W_out)
              Single output channel

    Process:
            1. Apply padding to input
            2. Calculate output dimensions
            3. For each output position:
               - Extract 3D patch from input
               - Compute element-wise product with kernel
               - Sum all products -> single output value
    """
    # Your code here
    ck, kd, kh, kw = kernel.shape
    input_c, input_d, input_h, input_w = input_volume.shape
    pd, ph, pw = padding
    sd, sh, sw = stride
    padded_video = np.pad(
        input_volume, ((0, 0), (pd, pd), (ph, ph), (pw, pw)), mode="constant"
    )
    out_d = (input_d - kd + 2 * pd) // sd + 1
    out_h = (input_h - kh + 2 * ph) // sh + 1
    out_w = (input_w - kw + 2 * pw) // sw + 1
    output = np.zeros((1, out_d, out_h, out_w))
    for d in range(out_d):
        for h in range(out_h):
            for w in range(out_w):
                d_start = d * sd
                d_end = d_start + kd
                h_start = h * sh
                h_end = h_start + kh
                w_start = w * sw
                w_end = w_start + kw
                patch = padded_video[:, d_start:d_end, h_start:h_end, w_start:w_end]
                output[0, d, h, w] = np.sum(patch * kernel)
    return output
