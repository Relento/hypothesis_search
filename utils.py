import ast
from config import cfg
import signal
import os
import traceback
import sys
import importlib
import re
import tokenize
from io import BytesIO
from collections import deque
import copy
import numpy as np
from contextlib import redirect_stdout
import time

def extract_function_blocks(code_string):
    try:
        file = BytesIO(code_string.encode())
        tokens = deque(tokenize.tokenize(file.readline))
        lines = []
        while tokens:
            token = tokens.popleft()
            if token.type == tokenize.NAME and token.string == 'def':
                start_line, _ = token.start
                last_token = token
                while tokens:
                    token = tokens.popleft()
                    if token.type == tokenize.NEWLINE:
                        break
                    last_token = token
                if last_token.type == tokenize.OP and last_token.string == ':':
                    indents = 0
                    while tokens:
                        token = tokens.popleft()
                        if token.type == tokenize.NL:
                            continue
                        if token.type == tokenize.INDENT:
                            indents += 1
                        elif token.type == tokenize.DEDENT:
                            indents -= 1
                            if not indents:
                                break
                        else:
                            last_token = token
                lines.append((start_line - 1, last_token.end[0] + 1))
        
        function_blocks = {}
        for start, end in lines:
            pattern = r"def\s+(\w+)\s*\("
            match = re.match(pattern, code_string.split('\n')[start])
            assert match
            function_name = match.group(1)
            function_block = code_string.split('\n')[start:end]
            function_blocks[function_name] = (start, end, function_block)
    except Exception as e:
        print('extract_function_blocks error: {}'.format(e))
        function_blocks = {}

    return function_blocks

# this implementaiton is wrong!
def extract_function_blocks_wrong(script_str):
    # Parse the script string into an abstract syntax tree (AST)
    ast_tree = ast.parse(script_str)
    
    # Define a dictionary to store the function blocks
    function_blocks = {}
    
    # Traverse the AST and extract the function blocks
    for node in ast_tree.body:
        if isinstance(node, ast.FunctionDef):
            # Get the function name and the start and end line numbers
            function_name = node.name
            if function_name == 'place_pattern':
                import ipdb; ipdb.set_trace()
            start_line = node.lineno - 1
            end_line = node.body[-1].lineno
            
            # Extract the block for the function
            function_block = script_str.split('\n')[start_line:end_line]
            
            # Store the function block in the dictionary
            function_blocks[function_name] = (start_line, end_line, function_block)
    
    return function_blocks

def update_script(old_script_str, updated_script_str):
    try:
        old_function_blocks = extract_function_blocks(old_script_str)
    except Exception as e:
        print('update_script error: {}'.format(e))
        return updated_script_str

    new_function_blocks = extract_function_blocks(updated_script_str)
    added_function_blocks = {}
    for function_name in new_function_blocks:
        if function_name not in old_function_blocks:
            added_function_blocks[function_name] = new_function_blocks[function_name]
            continue
        start_line, end_line, function_block = old_function_blocks[function_name]
        old_function_blocks[function_name] = (start_line, end_line, new_function_blocks[function_name][2])
    
    # Reconstruct the script string
    new_script_str = ''
    lines = old_script_str.split('\n')

    offset = 0
    items = list(old_function_blocks.items())
    items.sort(key=lambda x: x[1][0])
    for i in range(len(items)):
        function_name = items[i][0]
        start_line, end_line, function_block = items[i][1]
        offset_this = len(function_block) - (end_line - start_line)
        lines = lines[:start_line+offset] + function_block + lines[end_line+offset:]
        offset += offset_this
    
    for k, (_, _, v) in added_function_blocks.items():
       lines = v + lines
    
    return '\n'.join(lines)

if __name__ == '__main__':
    script_str = """
def add(x, y):
    result = x + y
    return result

def multiply(x, y):
    result = x * y
    return result

test()
"""
    new_script_str = """

def multiply(x, y):
    def add(x, y):
        return x + y
    return x * y

def add(x, y):
    result = x + y + 1
    print(result)
    return result

def divide(x, y):
    return x / y
"""

    print(update_script(script_str, new_script_str))


def test_impl(task, impl):
    results = {'train': [], 'test': []}
    def timeout_handler(signum, frame):
        raise TimeoutError('timeout')
    signal.signal(signal.SIGALRM, timeout_handler)
    timeout = 5
    # Raise a TimeoutError if the function takes more than 5 seconds to finish
    headers = \
'''
from typing import List
import numpy as np
from scipy.ndimage import label, find_objects
from scipy.signal import convolve2d
from scipy import ndimage
from collections import Counter
import math
np.int = int
'''
    if cfg.generate_detailed_plan_and_tests:
        impl = impl.replace('from tests import test_example_input, test_example_output', '')
    if cfg.generate_detailed_plan_and_tests:
        # add the first training example to the code
        input_grid = np.array(task['train'][0]['input'])
        output_grid = np.array(task['train'][0]['output'])
        headers += f'test_example_input = np.{repr(input_grid)}\ntest_example_output = np.{repr(output_grid)}\n\n'

    impl_src = headers + '\n' + impl

    
    while os.path.exists('temp/impl.py.lock'):
        time.sleep(0.1)

    open('temp/impl.py.lock', 'w').close()
    with open('temp/impl.py', 'w') as f:
        f.write(impl_src)

    try:
        signal.alarm(timeout)
        if 'temp.impl' in sys.modules:
            del sys.modules['temp.impl']
        temp_impl = importlib.import_module('temp.impl')
        importlib.reload(temp_impl)
        # from typing import List
        # import numpy as np
        # from scipy.ndimage import label, find_objects
        # from scipy.signal import convolve2d
        # exec(impl, locals())
        for split in ['train', 'test']:
            for i, t in enumerate(task[split]):
                input_grid = copy.deepcopy(t['input'])
                if cfg.grid_array_type == 'numpy':
                    input_grid = np.array(input_grid)
                # output_pred = locals()['transform_grid'](input_grid)
                with redirect_stdout(None):
                    output_pred = temp_impl.transform_grid(input_grid)
                if cfg.grid_array_type == 'numpy':
                    output_pred = np.array(output_pred).tolist()
                output_pred = [[int(e) for e in row] for row in output_pred]
                correct = t['output'] == output_pred


                results[split].append({
                    'output': output_pred,
                    'success': correct
                })

    except Exception as e:

        for split in ['train', 'test']:
            len_split = len(task[split])
            output_shape = len(task[split][0]['output']), len(task[split][0]['output'][0])
            output_grid = [[-1 for _ in range(output_shape[1])] for _ in range(output_shape[0])]
            if len(results[split]) < len_split:
                results[split].extend([{
                    'output': output_grid,
                    'success': False
                } for _ in range(len_split - len(results[split]))])
        
        if len(results['train']) != len(task['train']) or len(results['test']) != len(task['test']):
            import ipdb; ipdb.set_trace()

        signal.alarm(0)
        def prepare_stacktrace_feedback():
            traceback_msg = traceback.format_exc()
            start_idx = -1
            for i, line in enumerate(traceback_msg.split('\n')):
                if 'temp/impl.py' in line:
                    start_idx = i
                    break
            lines = traceback_msg.split('\n')[start_idx:]
            # get the current root directory
            current_root = os.getcwd()
            path_mapping = {
                # mapping machine-specific paths to generic paths
                current_root: '/root',
                '/data2/rcwang/miniconda3/': '/envs',
                # add your own python package mapping here
                # an example:
                '/home/rcwang/miniconda3/envs/hypothesis_search/': '/envs',


            }
            for i, l in enumerate(lines):
                for k, v in path_mapping.items():
                    l = l.replace(k, v)
                lines[i] = l
                
            return '\n'.join(lines)


        stacktrace_feedback = prepare_stacktrace_feedback()
        os.remove('temp/impl.py.lock')
        return {
            'results': results,
            'error': str(e),
            'stacktrace': traceback.format_exc(),
            'stacktrace_feedback': stacktrace_feedback,
        }

    if len(results['train']) != len(task['train']) or len(results['test']) != len(task['test']):
        import ipdb; ipdb.set_trace()

    signal.alarm(0)
    os.remove('temp/impl.py.lock')
    return {
        'results': results,
        'error': None
    }