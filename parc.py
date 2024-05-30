from dataset import get_larc_dataset, get_dataset
from gpt import GPT
from prompt import task_prompt, make_messages, task_prompt_grid
from tqdm import tqdm
import time
import copy
import os
import json
import numpy as np
import traceback
import signal
import shutil
from pprint import pprint
from visualize import get_grid_str, get_two_grid_str, get_two_grid_diff
import pyperclip
from collections import Counter
from language_generation import language_generation
import sys
import importlib
from program_cluster import ProgramClusterer
from contextlib import redirect_stdout
from utils import update_script, test_impl

from config import cfg

def web_ui_interaction(prompt):
    print(prompt)
    pyperclip.copy(prompt)
    _ = input('prompt copied to clipboard. Press enter to continue after you copy the response to file `resources/impl.py`.')
    with open('resources/impl.py', 'r') as f:
        impl = f.read()
    rets = [{
        'raw': impl,
        'parsed': impl.split('\n')
    }]
    return rets

def check_grid(grid):
    h = len(grid)
    w = len(grid[0])
    for i in range(h):
        if len(grid[i]) != w:
            return False
    return True

class SolverWithFeedback:
    def __init__(self, gpt, initial_prompt, initial_impl, task, desc=None, raw_response=None):
        self.gpt = gpt
        self.initail_prompt = initial_prompt
        self.initial_impl = initial_impl
        self.impls = [self.initial_impl]
        self.responses = [raw_response]
        self.feedback_prompts = []
        self.results = []
        self.num_feedbacks = 0
        self.task = task
        if desc is None:
            desc = task['descriptions']
        self.desc = desc
    
    def initial_solve(self):
        # handling multiple calls to initial_solve
        if len(self.results) > 0:
            return

        if cfg.directly_output_grid:
            # hack: assume self.impls[-1] is the parsed value from gpt class in this setting
            pred_outputs = self.impls[-1]['parsed']
            error_message = None
            if not isinstance(pred_outputs, list):
                error_message = pred_outputs

            results = {
                'train':[{'output': [[-1]], 'success':False} for i in range(len(self.task['train']))],
                'test': [],
            }

            for i in range(len(self.task['test'])):
                if error_message is not None:
                    results['test'].append({'output':[[-1]], 'success':False})
                else:
                    gt_output = self.task['test'][i]['output']
                    pred_output = pred_outputs[i]
                    success = gt_output == pred_output
                    results['test'].append({'output':pred_output, 'success':success})

            self.results.append({
                'impl': self.impls[-1]['raw'],
                'results': results,
                'error': error_message,
                'num_pass_train': 0,
                'num_feedbacks': 0,
                'prmopt': self.initail_prompt,
                'desc': self.desc,
            })

            return

        self.results.append({
            'impl': self.initial_impl,
            **test_impl(self.task, self.initial_impl),
            'num_feedbacks': self.num_feedbacks,
            'prompt': self.initail_prompt,
            'desc': self.desc,
        })
        self.compute_pass_all_train()

    def solve_with_feedback(self):
        if cfg.directly_output_grid:
            return
        # gather all previous implementations and feedback
        # and ask model to output another implementation
        feedback_prompt = None
        if cfg.grid_array_type == 'list':
            array_type = 'List[List[int]]'
        else:
            assert cfg.grid_array_type == 'numpy'
            array_type = 'np.ndarray[int]'
        if self.results[-1]['error'] is not None:
            print('Error', self.results[-1]['stacktrace_feedback'])

            if self.results[-1]['error'] == 'timeout':
                error_str = 'Timeout: The function takes too long to finish. '
            else:
                error_str = 'Error: ' + self.results[-1]['stacktrace_feedback'] 

            if not cfg.generate_detailed_plan_and_tests:
                feedback_prompt = error_str + f'\n Please reply with the corrected transform_grid(input_grid: {array_type}) in Python and nothing else.'
            else:
                feedback_prompt = error_str + f'\n Explain what the problem is, describe it, and then fix whichever function caused this mistake. Decompose the function if necessary. The updated function(s) should be the only code in your response. Enclose your corrected function with ```python and ```'
        else:
            for i, t in enumerate(self.results[-1]['results']['train']):
                if not t['success']:
                    same_shape = np.array(self.task['train'][i]['output']).shape == np.array(t['output']).shape
                    if cfg.feedback_show_grid_diff and same_shape:
                        feedback_prompt = f'Example {i} is wrong. Input:\n{get_grid_str(self.task["train"][i]["input"])}\nOutput comparison (for each entry x-y, x is the expected output and y is your output):\n{get_two_grid_diff(self.task["train"][i]["output"], t["output"])}\n'
                    else:
                        feedback_prompt = f'Example {i} is wrong. Input:\n{get_grid_str(self.task["train"][i]["input"])}\nExpected output:\n{get_grid_str(self.task["train"][i]["output"])}\nYour output:\n{get_grid_str(t["output"])}\n'

                    if cfg.feedback_describe_difference:
                        feedback_prompt += 'Please first describe the difference between your output grid and the expected grid, and then use this information to correct the transform_grid function. You should enclose your corrected function with ```python and ```.'
                    elif cfg.feedback_revise_language:
                        feedback_prompt += 'Note that the hint may be inaccurate or ambiguous. First think about whether hint need to be revised based on the input-output pairs. If so, output the revised hint. Then correct the transform_grid function based on the feedback and the revised prompt. You should enclose your corrected function with ```python and ```.'
                    elif cfg.generate_detailed_plan_and_tests:
                        feedback_prompt += 'Explain what the problem is, describe it, and then fix whichever function caused this mistake. Decompose the function if necessary. The updated function(s) should be the only code in your response. Enclose your corrected function with ```python and ```.'
                    else:
                        feedback_prompt += f'Please reply with the corrected transform_grid(input_grid: {array_type}) in Python and nothing else.'
                        # feedback_prompt += f'Please reply with the corrected transform_grid(input_grid: {array_type}) in Python and nothing else. Note that the bottom-left of a grid is grid[-1, 0] and top-right is grid[0, -1].'
                    # only give the feedback of one task case
                    break

        # correct impl
        if feedback_prompt is None:
            return
        else:
            if cfg.hint_may_be_wrong and cfg.add_hint:
                feedback_prompt += ' Note that the hint may be inaccurate or wrong, it is not mandatory for you to adhere to it.'
        
        self.feedback_prompts.append(feedback_prompt)
        if cfg.use_web_ui:
            rets = web_ui_interaction(feedback_prompt)
        else:
            if cfg.generate_detailed_plan_and_tests:
                messages = make_messages(self.initail_prompt, self.responses, self.feedback_prompts)
            else:
                messages = make_messages(self.initail_prompt, self.impls, self.feedback_prompts)
            #pprint(messages)
            if cfg.feedback_model_name == 'same':
                cfg.feedback_model_name = cfg.model_name
            rets = self.gpt.generate(messages_or_prompt=messages, 
                                    num_completions=1, 
                                    max_tokens=cfg.max_tokens, 
                                    dry_run=False, 
                                    indented=False,
                                    indented_after_first_line=False,
                                    model_name=cfg.feedback_model_name,
                                    stop=None,
                                    temperature=cfg.temperature,
                                    stats_group_name='feedback',
                                    )
            #pprint(rets)
        if isinstance(rets, str):
            # print('Error', rets)
            return

        # In case there are extra texts before the implementation
        start_idx = 0
        if not cfg.generate_detailed_plan_and_tests:
            for idx, l in enumerate(rets[0]['parsed']):
                if l.startswith('def transform_grid'):
                    start_idx = idx
                    break

            end_idx = len(rets[0]['parsed'])
            for idx in range(start_idx, len(rets[0]['parsed'])):
                l = rets[0]['parsed'][idx]
                if l.strip() != '' and l[0] != ' ':
                    end_idx = idx
                    break

        impl = '\n'.join(rets[0]['parsed'][start_idx:])

        if cfg.generate_detailed_plan_and_tests:
            last_impl = self.impls[-1]
            impl = update_script(last_impl, impl)

        self.num_feedbacks += 1
        self.results.append({
            'impl': impl,
            **test_impl(self.task, impl),
            'num_feedbacks': self.num_feedbacks,
            'prompt': self.initail_prompt,
            'desc': self.desc,
        })
        self.compute_pass_all_train()

        self.impls.append(impl)
        self.responses.append(rets[0]['raw'])

    def compute_pass_all_train(self):
        r = self.results[-1]
        states = [s['success'] for s in r['results']['train']]
        r['num_pass_train'] = len([s for s in states if s])
    
    def compute_pass_all_tests(self):
        r = self.results[-1]
        states = [s['success'] for s in r['results']['test']]
        r['num_pass_test'] = len([s for s in states if s])
        r['pass_all_tests'] = r['num_pass_test'] == len(self.task['test'])

    def pass_all_train(self):
        return self.results[-1]['num_pass_train'] == len(self.task['train'])

    def get_results(self):
        self.compute_pass_all_train()
        self.compute_pass_all_tests()
        r = self.results[-1]
        r['task_id'] = self.task['task_info']['task_id']
        r['feedback_prmopts'] = self.feedback_prompts
        return self.results[-1]


def evaluate(gpt, dataset, example_tasks=None, dry_run=True):
    num_completions = cfg.num_completions
    max_tokens = cfg.max_tokens
    add_hint = cfg.add_hint
    model_name = cfg.model_name
    visualize = cfg.visualize

    if cfg.program_clustering:
        program_cluster = ProgramClusterer()

    results = []
    if cfg.use_generated_language and cfg.language_generation.read_from_file:
            with open(cfg.language_generation.read_from_file, 'r') as f:
                task_id2desc = json.load(f)
            cfg.description_only_transformation = True
                
    for task in tqdm(dataset):
        # in dry run we do not generate the language
        if cfg.use_generated_language:
            if cfg.language_generation.read_from_file and not dry_run:
                k = str(task['task_info']['task_id'])
                if k not in task_id2desc:
                    descs = []
                else:
                    descs = task_id2desc[k]
                # currently assume only transformation description
                descriptions = [{
                    'description_output': d,
                    'description_input': '',
                    'description_output_grid_size': '',

                    } for d in descs]

            if dry_run:
                # use desc in the dataset as a proxy
                if cfg.language_generation.manually_select_file == '':
                    num_completions_language = 1
                else:
                    # num_completions_language = cfg.language_generation.num_completions
                    num_completions_language = 1
                prompts = [task_prompt(task, add_hint=add_hint) for _ in range(num_completions_language)]
            else:
                if not cfg.language_generation.read_from_file:
                    rets = language_generation(gpt, task, dry_run=dry_run, example_tasks=example_tasks)[0]
                    empty_desc = {'description_input': '', 'description_output_grid_size': '', 'description_output': ''}
                    descriptions = [d['parsed'] if isinstance(d['parsed'], dict) else empty_desc for d in rets]
                if cfg.directly_output_grid:
                    assert len(descriptions) == 1
                    prompts = [task_prompt_grid(task, add_hint=add_hint, desc=desc) for desc in descriptions]
                else:
                    prompts = [task_prompt(task, add_hint=add_hint, desc=desc) for desc in descriptions]
        else:
            if cfg.directly_output_grid:
                prompt = task_prompt_grid(task, add_hint=add_hint)
            else:
                prompt = task_prompt(task, add_hint=add_hint)

        rets_list = None
        solvers = []
        task_id = task['task_info']['task_id']

        def build_solver(rets, desc):
            solvers = []
            pass_all_train = False
            if isinstance(rets, str):
                return [SolverWithFeedback(gpt, prompt, {'parsed':[[[-1]]], 'raw': "[[[-1]]]"}, task, desc=desc, raw_response='Error')], False
            for impl_ in rets:
                if cfg.directly_output_grid:
                    impl = impl_
                else:
                    impl = '\n'.join(impl_['parsed'])
                    if not impl:
                        # print('empty impl', impl_['raw'])
                        continue
                solver = SolverWithFeedback(gpt, prompt, impl, task, desc=desc, raw_response=impl_['raw'])
                solvers.append(solver)
                solver.initial_solve()
                if solver.pass_all_train():
                    pass_all_train = True
                    break
            return solvers, pass_all_train
            
        pass_all_train = False
        if cfg.use_web_ui and not dry_run:
            rets = web_ui_interaction(prompt)
        else:
            if cfg.use_generated_language:
                rets_list = []
                for i, prompt in enumerate(prompts):
                    if pass_all_train:
                        break
                    #pprint(prompt)
                    rets = gpt.generate(messages_or_prompt=prompt, 
                                            num_completions=num_completions, 
                                            max_tokens=max_tokens, 
                                            dry_run=dry_run, 
                                            indented=False,
                                            indented_after_first_line=False,
                                            model_name=model_name,
                                            stop=None,
                                            temperature=cfg.temperature,
                                            stats_group_name='initial_solve'
                                            )
                    rets_list.append(rets)
                    #pprint(rets)
                    if not dry_run:
                        solvers_this, pass_all_train = build_solver(rets, descriptions[i])
                        solvers += solvers_this
            else:
                #print(prompt)
                rets = gpt.generate(messages_or_prompt=prompt, 
                                        num_completions=num_completions, 
                                        max_tokens=max_tokens, 
                                        dry_run=dry_run, 
                                        indented=False,
                                        indented_after_first_line=False,
                                        model_name=model_name,
                                        stop=None,
                                        temperature=cfg.temperature,
                                        stats_group_name='initial_solve'
                                        )
                #pprint(rets)
                if not dry_run:
                    solvers, pass_all_train = build_solver(rets, None)
                rets_list = [rets]
                prompts = [prompt]

        if dry_run:
            rets = rets_list[0]
            for i, rets_ in enumerate(rets_list):
                if i == 0:
                    continue
                for k in ['estimated_price', 'estimated_price_prompt', 'estimated_price_completion']:
                    rets[k] += rets_[k]
            results.append(rets)
        else:
            results_this = []

            if cfg.use_generated_language and not pass_all_train:
                # interleave the programs generated from different languages for minimizing the number of feedbacks
                n_language = len(rets_list)
                if len(solvers) != n_language * num_completions:
                    print('Some API calls failed only get {} programs'.format(len(solvers)))
                else:
                    solvers_old = solvers
                    solvers = []
                    for i in range(n_language * num_completions):
                        idx_language = i % n_language
                        solvers.append(solvers_old[idx_language * num_completions + i // n_language])
                    assert len(solvers) == len(solvers_old)

            if cfg.program_clustering and not pass_all_train:
                impls = [s.impls[0]for s in solvers]
                clusters = program_cluster.cluster(impls, required_task=task)
                idxs = [c[0]['idx'] for c in clusters.values()]
                # only select the first one in each cluster
                solvers = [s for i, s in enumerate(solvers) if i in idxs]
                print('num solvers', len(solvers))

            # solve with feedback until reaching maximum number of feedback
            # or all training examples are passed
            num_feedbacks = cfg.num_feedbacks
            selected_solver = None
            solved = False
            for i in range(cfg.num_feedbacks + 1):
                if solved:
                    break
                for solver in solvers:
                    if i == 0:
                        solver.initial_solve()
                    else:
                        solver.solve_with_feedback()
                    if solver.pass_all_train():
                        selected_solver = solver
                        solved = True
                        break


            if len(solvers) == 0:
                print('No prompt for', task['task_info']['task_id'])
                results.append(None)
                continue
            print(task['task_info']['task_id'], solved)
            if selected_solver is None:
                solvers.sort(key=lambda s: (s.get_results()['num_pass_train'], 1 if s.get_results()['error'] is None else 0), reverse=True)
                selected_solver = solvers[0]

            results_this = selected_solver.get_results()
            num_tests = len(task['test'])
            if cfg.directly_output_grid_vote:
                def np_array_vote(arrs):
                    import scipy.stats
                    arr = np.stack(arrs, axis=0)
                    output = scipy.stats.mode(arr, axis=0).mode[0]
                    return output

                success_solvers = list(filter(lambda s: s.get_results()['error'] is None, solvers))
                num_correct = 0
                for i in range(num_tests):
                    outputs = [s.get_results()['results']['test'][i]['output'] for s in success_solvers]
                    outputs = list(filter(lambda o: check_grid(o), outputs))
                    outputs = [np.array(o) for o in outputs]
                    if len(outputs) == 0:
                        pass
                    else:
                        size_vote = Counter([np.array(o).shape for o in outputs]).most_common()[0][0]
                        outputs_size = [o for o in outputs if o.shape == size_vote]
                        if len(outputs_size) == 1:
                            output_vote = outputs_size[0]
                        else:
                            output_vote = np_array_vote(outputs_size)
                        results_this['results']['test'][i]['output'] = output_vote.astype(int).tolist()
                        output_gt = np.array(task['test'][i]['output'])
                        success = None
                        if output_gt.shape == output_vote.shape:
                            success = bool((output_vote == output_gt).all())
                        else:
                            success = False
                        results_this['results']['test'][i]['success'] = success
                        if results_this['results']['test'][i]['success']:
                            num_correct += 1
                results_this['pass_all_tests'] = num_correct == num_tests

            # select the one that passes most training examples as the solution
            if cfg.use_generated_language:
                results_this['all_results'] = [s.get_results() if s != selected_solver and len(s.results) > 0 else None for s in solvers]
            results.append(results_this)
    
    if not dry_run:
        results_dir = cfg.results_dir
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        with open(os.path.join(results_dir, 'config.yaml'), 'w') as f:
            f.write(cfg.dump())
        shutil.copy('prompt.py', os.path.join(results_dir, 'prompt.py'))
        shutil.copy('parc.py', os.path.join(results_dir, 'parc.py'))
        shutil.copy('gpt.py', os.path.join(results_dir, 'gpt.py'))
        
        if visualize:
            from visualize import draw_grid
            from html_table import HTMLTableVisualizer, HTMLTableColumnDesc
            import textwrap
            table = HTMLTableVisualizer(os.path.join(results_dir, 'vis'), 'Visualization')
            table.begin_html()
            
            table.begin_table(f'Config', [
                HTMLTableColumnDesc('name', 'Name', 'code', {}),
                HTMLTableColumnDesc('content', 'Content', 'code', {}),
            ])
            for k, v in cfg.items():
                if k == 'openai_api_key':
                    continue
                table.row(name=k, content=str(v))
            table.end_table()
            for task, r in tqdm(zip(dataset, results)):
                if r is None:
                    continue
                task_id = r['task_id']
                table.begin_table(f'Task {task_id} Summary', [
                    HTMLTableColumnDesc('name', 'Name', 'code', {}),
                    HTMLTableColumnDesc('content', 'Content', 'code', {}),
                ])
                table.row(name='task_id', content=task_id)
                desc = task['descriptions']
                def wrap(text, width=50):
                    return '\n'.join(textwrap.wrap(text, width))

                table.row(name='description_input', content=wrap(desc['description_input']))
                table.row(name='description_output_grid_size', content=wrap(desc['description_output_grid_size']))
                table.row(name='description_output', content=wrap(desc['description_output']))
                table.row(name='implementation', content=r['impl'])
                table.row(name='num_pass_train_selected: ' + str(r['num_pass_train']), content='')
                table.row(name='pass_all_training: ' + str(r['num_pass_train'] == len(task['train'])), content='')
                table.row(name='pass_all_tests: ' + str(r['pass_all_tests']), content='')
                table.row(name='number of feedbacks used: ' + str(r['num_feedbacks']), content='')
                if cfg.dataset == 'arc1d':
                    for k, v in task['task_info']['results'].items():
                        table.row(name='1D-ARC ' + k + ': ' + str(v), content='')
                table.end_table()

                if r is None:
                    continue
                def build_solution_table(r, solution_alias):
                    table.begin_table(f'Task {task_id} Solutions Meta Info {solution_alias}', [
                        HTMLTableColumnDesc('name', 'Name', 'code', {}),
                        HTMLTableColumnDesc('content', 'Content', 'code', {}),
                    ])
                    table.row(name='description_input', content=wrap(r['desc']['description_input']))
                    table.row(name='description_output_grid_size', content=wrap(r['desc']['description_output_grid_size']))
                    table.row(name='description_output', content=wrap(r['desc']['description_output']))
                    table.row(name='implementation', content=r['impl'])
                    table.row(name='num_pass_train: ' + str(r['num_pass_train']), content='')
                    table.row(name='pass_all_training: ' + str(r['num_pass_train'] == len(task['train'])), content='')
                    table.row(name='pass_all_tests: ' + str(r['pass_all_tests']), content='')
                    table.row(name='number of feedbacks used: ' + str(r['num_feedbacks']), content='')

                    table.end_table()
                    table.begin_table(f'Task {task_id} Solutions Predictions {solution_alias}', [
                        HTMLTableColumnDesc('info', 'Info', 'code', {}),
                        HTMLTableColumnDesc('input_grid', 'Input', 'image', {'width': '200px'}),
                        HTMLTableColumnDesc('gt', 'GT', 'image', {'width': '200px'}),
                        HTMLTableColumnDesc('pred', 'Pred', 'image', {'width': '200px'}),
                    ])
                    for split in ['train', 'test']:
                        for i, t in enumerate(task[split]):
                            input_grid = t['input']
                            output_gt = t['output']
                            if len(r['results'][split]) <= i:
                                import ipdb; ipdb.set_trace()
                            output_pred = r['results'][split][i]['output']
                            try:
                                if np.array(output_pred).size == 0:
                                    output_pred = [[-1]]
                            except:
                                output_pred = [[-1]]
                            grid_img = draw_grid(input_grid)
                            grid_gt = draw_grid(output_gt)
                            if cfg.directly_output_grid and split == 'train':
                                grid_pred = draw_grid([[-1]])
                            else:
                                grid_pred = draw_grid(output_pred)
                            info = f'{split}-{i}\ncorrect: {r["results"][split][i]["success"]}\n'
                            # Check if the prediction is a valid grid
                            error = -1 in np.array(output_pred)
                            w = len(output_pred[0])
                            for row in output_pred:
                                if len(row) != w:
                                    error = True
                            info += f'error: {error}'

                            table.row(info=info, 
                                    input_grid=grid_img, gt=grid_gt, pred=grid_pred)
                    table.end_table()
                
                build_solution_table(r, 'Selected')
                if cfg.use_generated_language and cfg.language_generation.visualize_every_solution:
                    for idx, r_ in enumerate(r['all_results']):
                        if r_ is None:
                            continue
                        build_solution_table(r_, str(idx))
            current_root = os.getcwd()
            print('Visualization webpage dumped to ', os.path.join(current_root, cfg.results_dir, 'vis/index.html'))

    return results


        
if __name__ == '__main__':
    if cfg.use_web_ui:
        cfg.num_completions = 1
        cfg.num_feedbacks = 100

    model_name = cfg.model_name
    dataset = get_dataset(cfg.dataset)
    if cfg.flip_grid:
        for task in dataset:
            for split in ['train', 'test']:
                for t in task[split]:
                    t['input'] = [row[::-1] for row in t['input']]
                    t['output'] = [row[::-1] for row in t['output']]

    gpt = GPT(openai_key=cfg.openai_api_key)

    results = evaluate(gpt, dataset)

    # filter out tasks that exceed the context length
    filtered_dataset = []
    filtered_results = []
    for i, task in enumerate(dataset):
        if results[i]['exceed_context_length']:
            continue

        filtered_dataset.append(task)
        filtered_results.append(results[i])
    
    # print('Total number of tasks', len(filtered_dataset))

    if cfg.load_single_task < 0:
        if len(cfg.load_tasks) > 0:
            tasks_ids = cfg.load_tasks
        elif cfg.split_path != '':
            with open(cfg.split_path, 'r') as f:
                tasks_ids = json.load(f)
        else:
            tasks_ids = list(range(len(filtered_dataset)))
    else:
            tasks_ids = [cfg.load_single_task]

    idxs = [i for i, e in enumerate(filtered_dataset) if e['task_info']['task_id'] in tasks_ids]
    if cfg.num_examples > 0:
        idxs = idxs[cfg.start_idx:cfg.start_idx + cfg.num_examples]

    filtered_dataset = [filtered_dataset[i] for i in idxs]
    filtered_results = [filtered_results[i] for i in idxs]
    # Load examples for language generation
    example_tasks = None
    if cfg.language_generation.example_dataset != cfg.dataset:
        example_dataset = get_dataset(cfg.language_generation.example_dataset)
    else:
        example_dataset = dataset
    if cfg.use_generated_language:
        example_tasks = [task for task in example_dataset if task['task_info']['task_id'] in cfg.language_generation.example_task_ids]

    print('Task ids:', [task['task_info']['task_id'] for task in filtered_dataset])
    print('Total number of tasks', len(filtered_results))
    print(f'Estimated total cost:{sum([r["estimated_price"] for r in filtered_results]):.4f}')
    print(f'Estimated prompt cost:{sum([r["estimated_price_prompt"] for r in filtered_results]):.4f}')
    print(f'Estimated completion cost:{sum([r["estimated_price_completion"] for r in filtered_results]):.4f}')

    if os.getenv('SKIP_PRESS_Y', '0') == '1':
        inp = 'Y'
    else:
        inp = input('Press Y to continue...')

    if inp == 'Y':
        results = evaluate(gpt, filtered_dataset, example_tasks=example_tasks, dry_run=False)
        # filter out invalid results
        results = [r for r in results if r is not None]
        print('Task ids:', [task['task_info']['task_id'] for task in filtered_dataset])
        print('Total number of tasks', len(filtered_dataset))
        print('Number of invalid tasks:', len(filtered_dataset) - len(results))
        print('Number of tasks that pass all tests', len([r for r in results if r['pass_all_tests']]))
        # total # of tests
        print('All tasks that passed:', [r['task_id'] for r in results if r['pass_all_tests']])
        gpt.print_summary()

