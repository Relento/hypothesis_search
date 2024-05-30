from dataset import get_larc_dataset
import signal
import importlib
import copy
from config import cfg
import numpy as np
from tqdm import tqdm
from contextlib import redirect_stdout
import traceback
import multiprocessing as mp

def evaluate_program(impl, task):
    results = {'train': [], 'test': []}
    def timeout_handler(signum, frame):
        raise TimeoutError('timeout')
    signal.signal(signal.SIGALRM, timeout_handler)
    timeout = 1
    # Raise a TimeoutError if the function takes more than 5 seconds to finish
    signal.alarm(timeout)

    try:
        exec(impl, locals())
        from typing import List
        import numpy as np
        from scipy.ndimage import label, find_objects
        from scipy.signal import convolve2d
        for split in ['train', 'test']:
            for i, t in enumerate(task[split]):
                input_grid = copy.deepcopy(t['input'])
                if cfg.grid_array_type == 'numpy':
                    input_grid = np.array(input_grid)
                with redirect_stdout(None):
                    output_pred = locals()['transform_grid'](input_grid)
                if cfg.grid_array_type == 'numpy':
                    output_pred = np.array(output_pred).tolist()
                output_pred = [[int(e) for e in row] for row in output_pred]


                results[split].append({
                    'output': output_pred,
                })

    except Exception as e:
        signal.alarm(0)
        return {
            'results': results,
            #TODO: decide if we should inlcude this.
            'error': str(e),
            # 'stacktrace': traceback.format_exc()
        }

    if len(results['train']) != len(task['train']) or len(results['test']) != len(task['test']):
        import ipdb; ipdb.set_trace()

    signal.alarm(0)
    return {
        'results': results,
        'error': None
    }

        
class ProgramClusterer:
    def __init__(self, ):
        # only use 50 tasks to evlauate
        n = 0
        if n == 0:
            self.dataset = []
        else:
            self.dataset = get_larc_dataset()[:n]
        
        num_workers = 16
        self.pool = mp.Pool(num_workers)
    
    def cluster(self, programs, required_task=None):
        results_all = []
        print('getting results', len(programs))
        tasks = self.dataset
        if required_task is not None:
            tasks = [required_task] + tasks
        results_all = []
        multiprocessing = True
        if multiprocessing:
            results_all = self.pool.starmap(evaluate_program, [(program, task) for program in programs for task in tasks])
        else:
            results_all = []
            for program in programs:
                results = []
                tasks = self.dataset
                if required_task is not None:
                    tasks = [required_task] + tasks
                for task in tasks:
                    results.append(evaluate_program(program, task))
                results_all.append(results)

        print(len(results_all))
        
        clusters = {}
        print('clustering')
        for i in tqdm(range(len(programs))):
            key = str(results_all[i])
            if key not in clusters:
                clusters[key] = []
            clusters[key].append({
                'impl': programs[i], 
                'idx': i,
            })
        # print('cluster stats:', [len(v) for v in clusters.values()])
        return clusters
        
    
    def compare_results(results1, results2):
        for split in ['train', 'test']:
            if len(results1[split]) != len(results2[split]):
                return False
            for i in range(len(results1[split])):
                output1 = results1[split][i]['output']
                output2 = results2[split][i]['output']
                if results1[split][i]['success'] != results2[split][i]['success']:
                    return False
                elif (np.array(output1) != np.array(output2)).all():
                    return False
        
        return results1['error'] == results2['error']

