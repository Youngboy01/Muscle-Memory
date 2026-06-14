import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape
    out_height = (input_height-kernel_height+2*padding)//stride+1
    out_width = (input_width-kernel_width+2*padding)//stride +1 
    padded_matrix = np.pad(input_matrix,((padding,padding),(padding,padding)),mode='constant')
    output_matrix = np.zeros((out_height,out_width))
    for h in range(out_height):
        for w in range(out_width):
            h_start = h*stride 
            w_start = w*stride
            h_end = h_start+kernel_height
            w_end = w_start+kernel_width
            patch = padded_matrix[h_start:h_end,w_start:w_end]
            output_matrix[h,w] = np.sum(patch*kernel)
    # Your code here
    return output_matrix
