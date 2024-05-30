from gpt import GPT
from config import cfg
from visualize import get_grid_str
from consts import COLORS
from pprint import pprint
import os
import json
from tqdm import tqdm
import numpy as np


def task_prompt(task, add_hint=True, is_example=False):
    messages = []
    user_prompt = ''
    for i, t in enumerate(task['train']):
        user_prompt += f'Case {i}:\nInput:\n'
        user_prompt += get_grid_str(t['input']) + '\n'
        user_prompt += 'Output:\n'
        user_prompt += get_grid_str(t['output']) + '\n'
    
    messages.append({'content': user_prompt, 'role': 'user'})
    

    if add_hint:
        assist_prompt = ''
        desc = task['descriptions']
        hint = \
f'''Describing the input grid: {desc["description_input"].strip().replace('...', ' ')}
Describing the size of the output grid: {desc["description_output_grid_size"].strip().replace('...', ' ')}
Describing how to transform the grid: {desc["description_output"].strip().replace('...', ' ')}
'''
        assist_prompt += hint
        messages.append({'content': assist_prompt, 'role': 'assistant'})
    return messages

def language_generation_prompt(task, example_tasks=None):
    # assert len(example_tasks) > 0
    messages = []

    system_prompt = 'You will be given a list of input-output pairs. Each input and output is a grid of numbers representing representing a visual grid. There is a SINGLE pattern that transforms each input grid to the corresponding output grid.'
    if cfg.language_generation.add_meta_hint:
        system_prompt += \
'''
The pattern may involve counting or sorting objects (e.g. sorting by size), comparing numbers (e.g. which shape or symbol appears the most? Which is the largest object? Which objects are the same size?), or repeating a pattern for a fixed number of time.
There are other concepts that may be relevant.
- Lines, rectangular shapes
- Symmetries rotations, translations.
- Shape upscaling or downscaling, elastic distortions.
- Containing / being contained / being inside or outside of a perimeter.
- Drawing lines, connecting points, orthogonal projections.
- Copying, repeating objects.
You should treat black cells as empty cells (backgrounds).
'''
    system_prompt += '\nThe number in the input grid can be mapped to the following colors:' + '; '.join([f"{c}:{COLORS[c]['color_name']}" for c in range(10)])
    system_prompt += '\nOutput the language description of the transformation.'

    if len(example_tasks) ==  0:
        system_prompt += 'Your description should be in the format:\nDescribing the input grid:{text}\n Describing the size of the output grid:{text}\n Describing how to transform the grid:{text}\n'
    messages.append({'content': system_prompt, 'role': 'system'})
    for example_task in example_tasks:
        messages.extend(task_prompt(example_task, add_hint=True, is_example=True))
    
    messages.extend(task_prompt(task, add_hint=False))
    return messages

def language_generation(model, dataset, dry_run=True, example_tasks=None):
    if not isinstance(dataset, list):
        dataset = [dataset]

    if cfg.language_generation.manually_select_file != '':
        with open(cfg.language_generation.manually_select_file, 'r') as f:
            manually_selected = json.load(f)

    results = []
    for i, task in tqdm(enumerate(dataset), total=len(dataset)):
        messages = language_generation_prompt(task, example_tasks=example_tasks)
        if not dry_run:
            pass

        def parse_response(text):
            try:
                lines = text.split('\n')
                lines = ' '.join(lines)
                lines = lines.split('Describing the size of the output grid:')
                grid_desc = lines[0].split(':')[-1].strip()
                size_desc, transform_desc = lines[1].split('Describing how to transform the grid:')
                size_desc = size_desc.strip()
                transform_desc = transform_desc.strip()
                return {
                    'description_input': grid_desc,
                    'description_output_grid_size': size_desc,
                    'description_output': transform_desc,
                }
            except:
                return "Error parsing response"


        results_this = model.generate(messages_or_prompt=messages, 
                                        num_completions=cfg.language_generation.num_completions, 
                                        max_tokens=cfg.language_generation.max_tokens, 
                                        rate_limit_tokens=4800,
                                        dry_run=dry_run, 
                                        indented=False,
                                        indented_after_first_line=False,
                                        model_name=cfg.language_generation.model_name,
                                        stop=None,
                                        temperature=cfg.language_generation.temperature,
                                        parsing_func=parse_response,
                                        stats_group_name='language_generation',
                                        )
        if cfg.language_generation.manually_select_file != '':
            if str(task['task_info']['task_id']) in manually_selected:
                idxs = manually_selected[str(task['task_info']['task_id'])]
                if not isinstance(idxs, list):
                    idxs = [idxs]
                results_this = [results_this[idx] for idx in idxs]

        results.append(results_this)
    return results

if __name__ == '__main__':
    from dataset import get_dataset
    import pyperclip

    model_name = cfg.model_name
    dataset = get_dataset(cfg.dataset)
    if cfg.language_generation.example_dataset != cfg.dataset:
        example_dataset = get_dataset(cfg.language_generation.example_dataset)
    else:
        example_dataset = dataset
    example_tasks = [task for task in example_dataset if task['task_info']['task_id'] in cfg.language_generation.example_task_ids]
    gpt = GPT(openai_key=cfg.openai_api_key)

    results = language_generation(gpt, dataset, dry_run=True, example_tasks=example_tasks)

    # filter out tasks that exceed the context length
    filtered_dataset = []
    filtered_results = []
    for i, task in enumerate(dataset):
        if results[i]['exceed_context_length']:
            continue
        filtered_dataset.append(task)
        filtered_results.append(results[i])
    
    print('Total number of tasks', len(filtered_dataset))

    if cfg.load_single_task < 0:
        if len(cfg.load_tasks) > 0:
            tasks_ids = cfg.load_tasks
        else:
            with open(cfg.split_path, 'r') as f:
                tasks_ids = json.load(f)
    else:
            tasks_ids = [cfg.load_single_task]
    
    idxs = [i for i, e in enumerate(filtered_dataset) if e['task_info']['task_id'] in tasks_ids]


    if cfg.num_examples > 0:
        idxs = idxs[cfg.start_idx:cfg.start_idx + cfg.num_examples]
    

    filtered_dataset = [filtered_dataset[i] for i in idxs if i not in cfg.language_generation.example_task_ids]
    filtered_results = [filtered_results[i] for i in idxs if i not in cfg.language_generation.example_task_ids]

    print('Total number of tasks', len(filtered_dataset))
    print('Tasks to be loaded:', len(idxs))
    print('Example tasks:', cfg.language_generation.example_task_ids)
    print(sum([r['estimated_price'] for r in filtered_results]))

    if os.getenv('SKIP_PRESS_Y', '0') == '1':
        inp = 'Y'
    else:
        inp = input('Press Y to continue...')

    if inp == 'Y':
        results = language_generation(gpt, filtered_dataset, dry_run=False, example_tasks=example_tasks)
        for i, (r, task) in enumerate(zip(results, filtered_dataset)):
            print(task['task_info']['task_id'])
            print(task['descriptions'])
            r = {'task_id':task['task_info']['task_id'], 'results': r}
            results[i] = r
        
        gpt.print_summary()
        
        with open('resources/generated_languages/gpt4_0613_100_64.json', 'w') as f:
            json.dump(results, f, indent=4)


