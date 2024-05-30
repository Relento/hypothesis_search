from PIL import Image, ImageDraw
from consts import COLORS
import numpy as np
np.set_printoptions(linewidth=1000)
from config import cfg
import re

def get_grid_str(grid):
    value = cfg.grid_value
    style = cfg.grid_style
    if value == 'number':
        pass
    elif value == 'color':
        grid = [[COLORS[c]['color_name'] for c in row] for row in grid]
    else:
        raise ValueError('Invalid value: {}'.format(value))
    if style is None:
        grid_str = '\n'.join([''.join([str(c) for c in row]) for row in grid])
    elif style == 'numpy':
        grid_str = str(np.array(grid))
        if cfg.add_coordinate:
            grid_ = np.array(grid).astype(str)
            h, w = grid_.shape
            for i in range(h):
                for j in range(w):
                    grid_[i, j] = grid_[i,j] + '({},{})'.format(i, j)
            grid_str = str(grid_)
            grid_str = grid_str.replace('\'', '')
        if value == 'color':
            grid_str = grid_str.replace("'", '')
        else:
            if cfg.add_space_first_number:
                # Define a regular expression pattern to match '[' followed by a number
                pattern = r'\[(\d)'
                # Use the re.sub() function to replace the matched pattern with the modified string
                grid_str = re.sub(pattern, r'[ \g<1>', grid_str)
        

    return grid_str

def get_two_grid_diff(grid_ref, grid_pred):
    assert cfg.grid_value != 'color'
    assert cfg.grid_style == 'numpy'
    # showing the difference between grid_ref and grid_pred
    # by printing the entries in grid_ref and grid_pred simultaneously
    grid_ref = np.array(grid_ref)
    grid_pred = np.array(grid_pred)
    assert grid_ref.shape == grid_pred.shape
    grid_diff = np.zeros(grid_ref.shape, dtype=object)
    for i in range(grid_ref.shape[0]):
        for j in range(grid_ref.shape[1]):
            grid_diff[i,j] = '{}-{}'.format(grid_ref[i,j], grid_pred[i,j])
    grid_str = str(grid_diff).replace("'", '')

    return grid_str


def get_two_grid_str(grid1, grid2, padding=2):
    w1, w2 = len(grid1[0]), len(grid2[0])
    h1, h2 = len(grid1), len(grid2)
    grid1_str = str(np.array(grid1)).split('\n')
    grid2_str = str(np.array(grid2)).split('\n')
    row_strs = []
    for i in range(max(h1, h2)):
        row_str = ''
        if i < h1:
            row_str += grid1_str[i]
        else:
            row_str += ' ' * (w1 * 2 + 2)
        row_str += ' ' * padding
        if i == h1 - 1:
            row_str = row_str[:-1]
        if i < h2:
            row_str += grid2_str[i]
        row_strs.append(row_str)

    return '\n'.join(row_strs)

def draw_grid(grid):
    # Define the size of each cell in pixels
    CELL_SIZE = 100

    # Define the grid of numbers
    # Calculate the size of the image based on the size of the grid
    width = len(grid[0]) * CELL_SIZE
    height = len(grid) * CELL_SIZE

    # Create a new image with a black background
    image = Image.new('RGB', (width, height), '#000')

    # Create a new ImageDraw object
    draw = ImageDraw.Draw(image)

    # Iterate over each cell in the grid
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            # Calculate the position of the cell in the image
            x = j * CELL_SIZE
            y = i * CELL_SIZE

            # Get the color for the cell based on its value
            if j <0 or len(grid[i]) <= j or not isinstance(grid[i][j], int) or grid[i][j] < 0 or grid[i][j] >= len(COLORS):
                color = '#FFFFFF'
            else:
                color = COLORS[grid[i][j]]['rgb']
            border_color = COLORS[5]['rgb'] # grey

            # Draw a rectangle with the appropriate color
            # image.paste(color, (x, y, x + CELL_SIZE, y + CELL_SIZE))
            draw.rectangle((x, y, x + CELL_SIZE - 1, y + CELL_SIZE - 1), fill=color, outline=border_color, width=1)

    return image


if __name__ == '__main__':
    grid = [
        [0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0],
        [0, 2, 0, 2, 0],
        [0, 0, 3, 4, 3],
        [0, 0, 0, 3, 0],
    ]
    image = draw_grid(grid)

    # Display the image
    image.show()
