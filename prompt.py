from visualize import get_grid_str, get_two_grid_str
from consts import COLORS
from config import cfg

def task_prompt(task, add_hint=True, desc=None):
    general_prompt = \
'''You will be given a list of input output pairs. Each input and output is a grid of numbers (from 0 to 9, int not str). Your job is to infer the python program that transforms the input grid to the corresponding output grid. The input-output pairs are given below:
'''
    prompt = general_prompt
    if cfg.grid_array_type == 'list':
        array_type = 'List[List[int]]'
    else:
        assert cfg.grid_array_type == 'numpy'
        array_type = 'np.ndarray[int]'

    for i, t in enumerate(task['train']):
        if cfg.example_side_by_side:
            prompt += f'Example {i}\n'
            prompt += get_two_grid_str(t['input'], t['output']) + '\n'
        else:
            prompt += f'Example {i}:\nInput:\n'
            prompt += get_grid_str(t['input']) + '\n'
            prompt += 'Output:\n'
            prompt += get_grid_str(t['output']) + '\n'
    
    prompt += f'Now, please write a python program transform_grid(input_grid: {array_type}) -> {array_type} that transforms the input grid to the corresponding output grid.\n'
    if cfg.add_meta_hint:
        prompt += \
f'''\nThe transformation may involve counting or sorting objects (e.g. sorting by size), comparing numbers (e.g. which shape or symbol appears the most? Which is the largest object? Which objects are the same size?), or repeating a pattern for a fixed number of time.

There are other concepts that may be relevant.
- Lines, rectangular shapes
- Symmetries rotations, translations.
- Shape upscaling or downscaling, elastic distortions.
- Containing / being contained / being inside or outside of a perimeter.
- Drawing lines, connecting points, orthogonal projections.
- Copying, repeating objects.
You should treat black cells as empty cells (backgrounds).'''

    floodfill = False
    if floodfill:
        prompt += 'You may assume an additional helper function floodfill(grid, x, y, color) that fills all the connected cells from cell (x, y) with the given color (you DO NOT need to implement it and you may not need this function). \n'

    if add_hint:
        if cfg.hint_may_be_wrong:
            prompt += 'Hint (may be wrong): You may want to use the following hint to implement the function, but it maybe inaccurate or wrong. So you should focus more on the given input-output examples: \n'
        else:
            prompt += 'Hint: You may want to use the following guidance to implement the function: \n'
        # currently only use the first description
        if desc is None:
            desc = task['descriptions']
        if cfg.description_only_transformation:
            hint = (desc['description_output'].strip() + '.').replace('...', ' ').replace('..', '.')
        else:
            if not isinstance(desc, dict):
                print('Description is not a dict, using empty description', desc)
                desc = {'description_input': '', 'description_output_grid_size': '', 'description_output': ''}
            hint = (desc['description_input'].strip() + '. ' + desc['description_output_grid_size'].strip() + '. ' + desc['description_output'].strip()).replace('...', ' ').replace('..', '.')
        prompt += hint
        prompt += '\nThe number in the input grid can be mapped to the following colors:' + '; '.join([f"{c}:{COLORS[c]['color_name']}" for c in range(10)])
    
    if cfg.add_scipy_hint:
        prompt += '\nAnother hint: the problem may involve recognizing shapes in the grid. Some useful functions are: scipy.ndimage.label, scipy.ndimage.find_objects. You should AVOID memorizing shapes in the training examples unless necessary and use find_objects more often. An example usage is:'
        prompt += \
'''
```python
# extract the bounding boxes of all the shapes with color blue (1)

# criteria for determining whether two pixel belong to one single shape
# by default two pixels are connected when they only differ by 1 in row or column coordinate
structure = None 
# sometimes treat two pixels as connected when they are only diagonally connected
structure = np.ones((3, 3)) 
label_grid, _ = scipy.ndimage.label(grid == 1, structure=structure))
objects = scipy.ndimage.find_objects(label_grid)
for obj in objects:
    shape = grid[obj]
    ...
```'''

    if cfg.generate_language_before_program:
        prompt += \
f'''\nThe transformation may involve counting or sorting objects (e.g. sorting by size), comparing numbers (e.g. which shape or symbol appears the most? Which is the largest object? Which objects are the same size?), or repeating a pattern for a fixed number of time.
There are other concepts that may be relevant.
- Lines, rectangular shapes
- Symmetries rotations, translations.
- Shape upscaling or downscaling, elastic distortions.
- Containing / being contained / being inside or outside of a perimeter.
- Drawing lines, connecting points, orthogonal projections.
- Copying, repeating objects.
You should treat black cells as empty cells (backgrounds).

First try to describe the transformation in natural language in the following format:
Describing the input grid: {{text}}
Describing the size of the output grid: {{text}}
Describing how to transform the grid: {{text}}

Then reply with the implementation of transform_grid(input_grid: {array_type}) enclose by ```python and ```. Each cell in the output should only be numbers from 0 to 9.'''
    elif cfg.add_hint and cfg.generate_detailed_plan_and_tests:
        #prompt += f'\n First output a detailed plan by translate the hint to be more procedural (step by step), then implement the plan as a python function, "transform_grid" enclosed by ```python and ```. Expect to take a numpy array as input. Write as many helper functions as needed. Immediately after each helper function, write a test for that helper function. Keep your tests concise and minimal, and ensure they print out helpful information if they fail. Reference the Example 0 by including `from tests import test_example_input, test_example_output` at the top of your code. First reply with the detailed plan, followed by the python function "transform_grid" enclosed by ```python and ``` in a single response.'
        prompt += f'\n Implement the python function, "transform_grid" enclosed by ```python and ``` according to the hint. Expect to take a numpy array as input. Write as many helper functions as needed. Immediately after each helper function, write a test for that helper function. Keep your tests concise and minimal, and ensure they print out helpful information if they fail. Reference the Example 0 by including `from tests import test_example_input, test_example_output` at the top of your code.'
    else:
        prompt += f'\nJust reply with the implementation of transform_grid(input_grid: {array_type}) in Python and nothing else, each cell in the output should only be numbers from 0 to 9.'
    return prompt

def task_prompt_grid(task, add_hint=True, desc=None):
    general_prompt = 'You will be given a list of input-output pairs. Each input and output is a grid of numbers. There is a single pattern that transforms each input grid to the corresponding output grid. You will be given a new input grid, and your job is to infer its corresponding output grid with the same transformation. The example input-output pairs are given below:'
    prompt = general_prompt
    for i, t in enumerate(task['train']):
        if cfg.example_side_by_side:
            prompt += f'Example {i}\n'
            prompt += get_two_grid_str(t['input'], t['output']) + '\n'
        else:
            prompt += f'Example {i}:\nInput:\n'
            prompt += get_grid_str(t['input']) + '\n'
            prompt += 'Output:\n'
            prompt += get_grid_str(t['output']) + '\n'
    if add_hint:
        prompt += 'Hint: You may want to use the following guidance to implement the function: \n'
        # currently only use the first description
        if desc is None:
            desc = task['descriptions']
        hint = (desc['description_input'].strip() + '. ' + desc['description_output_grid_size'].strip() + '. ' + desc['description_output'].strip()).replace('...', ' ').replace('..', '.')
        prompt += hint
        prompt += '\nThe number in the input grid can be mapped to the following colors:' + '; '.join([f"{c}:{COLORS[c]['color_name']}" for c in range(10)])
    n = len(task['test'])
    if n == 1:
        prompt += 'Now, please output the grid for the following input and nothing else:\n'
    else:
        assert n > 1
        prompt += 'Now, please output the grids for the following input grids and nothing else:\n'
    for i, t in enumerate(task['test']):
        prompt += f'Test Example {i}:\n'
        prompt += get_grid_str(t['input']) + '\n'
    prompt += 'Please reply with the format:\nTest Output 0:\n{matrix}\n...'
    return prompt

def make_messages(initial_prompt, impls, feedbacks):
    messages = []
    messages.append({"role": "system", "content": ""})
    messages.append({"role": "user", "content": initial_prompt})
    if cfg.only_last_feedback:
        impls = impls[-1:]
        feedbacks = feedbacks[-1:]
    for impl, feedback in zip(impls, feedbacks):
        messages.append({"role": "assistant", "content": impl})
        messages.append({"role": "user", "content": feedback})
    return messages


if __name__ == '__main__':
    from dataset import get_larc_dataset
    import pyperclip
    dataset = get_larc_dataset()
    tasks_ids = [29]
    idxs = [i for i, e in enumerate(dataset) if e['task_info']['task_id'] in tasks_ids]
    add_hint = True
    for idx in idxs:
        task = dataset[idx]
        prompt = task_prompt(task, add_hint=add_hint)
        pyperclip.copy(prompt)
        print(prompt)
