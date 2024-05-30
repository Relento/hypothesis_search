
from typing import List
import numpy as np
from scipy.ndimage import label, find_objects
from scipy.signal import convolve2d
from scipy import ndimage
from collections import Counter
import math
np.int = int

def transform_grid(input_grid: np.ndarray) -> np.ndarray:
    size = input_grid.shape[1] * 2 - 1
    output_grid = np.zeros((size, size), dtype=int)
    
    for i in range(input_grid.shape[1]):
        output_grid[i:size-i, i] = input_grid[0,i]

    return np.fliplr(output_grid)