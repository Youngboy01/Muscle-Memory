import numpy as np
import math


def sobel_edge_detection(image):
    """
    Apply Sobel edge detection to a grayscale image.

    Args:
        image: 2D list/array representing a grayscale image
               with values in range [0, 255]

    Returns:
        Edge magnitude image as 2D list with integer values (0-255),
        or -1 if input is invalid
    """
    # Your code here
    m = len(image)
    n = len(image[0])
    if any(pixel > 255 for row in image for pixel in row) or m < 3 or n < 3:
        return -1
    Gxk = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Gyk = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    magnitudes = []
    for r in range(m - 2):
        row_mag = []
        for c in range(n - 2):
            Gx = 0
            Gy = 0
            for i in range(3):
                for j in range(3):
                    pixel = image[r + i][c + j]
                    Gx += pixel * Gxk[i][j]
                    Gy += pixel * Gyk[i][j]

            mag = math.sqrt(Gx**2 + Gy**2)
            row_mag.append(mag)

        magnitudes.append(row_mag)

    max_mag = max(max(row) for row in magnitudes)

    if max_mag == 0:
        return [[0] * (n - 2) for _ in range(m - 2)]

    output = []
    for row in magnitudes:
        output.append([int((val / max_mag) * 255) for val in row])

    return output
