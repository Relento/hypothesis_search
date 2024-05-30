import os
from config import cfg
import pickle
import pandas as pd
from tqdm import tqdm
import json
from collections import defaultdict


def get_dataset(dataset):
    if dataset == 'larc':
        return get_larc_dataset()
    elif dataset == 'arc1d':
        return get_arc1d_dataset()
    else:
        raise NotImplementedError

def get_larc_dataset():
    cache_path = cfg.larc_cache_path
    only_load_success_task = cfg.only_load_success_task
    if only_load_success_task:
        cache_path = cache_path.replace('.pkl', '_success.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            tasks = pickle.load(f)
    
    else:
        data_path = cfg.larc_dataset_path
        summary_path = os.path.join(data_path, 'summary')
        task_path = os.path.join(data_path, 'tasks_json')

        keys = ['task', 'join', 'build', 'description']
        df_dict = {}
        for k in keys:
            df_dict[k] = pd.read_csv(os.path.join(summary_path, k + '.csv'))

        ct = 0
        success = 0
        tasks = {}

        for _, row in tqdm(df_dict['join'].iterrows()):
            task_id, description_id, build_id = row['task_id'], row['description_id'], row['build_id']
            task = df_dict['task'][df_dict['task']['task_id'] == task_id].iloc[0]
            task = task.to_dict()
            description = df_dict['description'][df_dict['description']['description_id'] == description_id].iloc[0]
            description = description.to_dict()
            # check if build id is nan
            if pd.isna(build_id):
                build = None
            else:
                build = df_dict['build'][df_dict['build']['build_id'] == build_id].iloc[0]
                build = build.to_dict()
            
            description['build'] = build

            ct += 1
            if build is not None and description['is_verified']:
                if build['is_success']:
                    success += 1
                elif only_load_success_task:
                    continue
                if task_id not in tasks:
                    with open(os.path.join(task_path, f'{task_id}.json')) as f:
                        task_json = json.load(f)
                    tasks[task_id] = {
                        'task_info': task,
                        'descriptions': [description],
                        'train': task_json['train'],
                        'test': task_json['test'],
                    }
                else:
                    tasks[task_id]['descriptions'].append(description)

        tasks = list(tasks.values())
        print(f'{success} / {ct} = {success / ct}')
        print(f'task with at least one success: {len(tasks)} / {len(df_dict["task"])} = {len(tasks) / len(df_dict["task"])}')

        with open(cache_path, 'wb') as f:
            pickle.dump(tasks, f)

    with open(cfg.description_id_override_json, 'r') as f:
        description_id_override_d = json.load(f)

    for task in tasks:
        # select a given description
        task_id_str = str(task['task_info']['task_id'])
        if cfg.description_id_override and task_id_str in description_id_override_d:
            found = False
            for desc in task['descriptions']:
                if desc['description_id'] == description_id_override_d[task_id_str]:
                    task['descriptions'] = desc
                    found = True
                    break
            if not found:
                print(f'Could not find description_id {description_id_override_d[task_id_str]} for task {task_id_str}')
        else:
            # select the description with the highest success rate
            if cfg.description_select_by_heuristic:
                desc_dict = defaultdict(list)
                for desc in task['descriptions']:
                    build = desc['build']
                    if build is None:
                        continue
                    numerator = build['is_success']
                    denom = build['num_attempts']
                    desc_dict[desc['description_id']].append(numerator / denom)

                desc_dict = {k: sum(v) / len(v) for k, v in desc_dict.items()}
                max_desc_id = max(desc_dict, key=desc_dict.get)
                for i, desc in enumerate(task['descriptions']):
                    if desc['description_id'] == max_desc_id:
                        task['descriptions'] = desc
                        break
            else:
                task['descriptions'] = task['descriptions'][0]

    return tasks

def get_arc1d_dataset():
    cache_path = cfg.arc1d_cache_path
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            tasks = pickle.load(f)
        return tasks
    else:
        data_path = cfg.arc1d_dataset_path
        mapping_path = os.path.join(data_path, 'mapping.json')
        with open(mapping_path, 'r') as f:
            mapping_dict = json.load(f)

        tasks = []
        ct = 1
        sampled_idxs = []
        ct_this = 0

        # for retrieving the results from the original paper
        direct_grid_results_path = '../LLM4ARC/output-logs/direct-grid/1D-ARC'
        object_based_results_path = '../LLM4ARC/output-logs/object-based/1D-ARC'

        for concept, path in mapping_dict.items():
            path_this = os.path.join(data_path, 'dataset', path)
            files = sorted(os.listdir(path_this))
            ct_this = 0

             
            def sort_func(s):
                return s.apply(lambda s: int(s.split('_')[-1]))
            direct_grid_csv_file = os.path.join(direct_grid_results_path, f'{path}.json')
            direct_grid_df = pd.read_csv(direct_grid_csv_file, delimiter=',')
            direct_grid_df = direct_grid_df[direct_grid_df['GPT_version'] == 4]
            direct_grid_df.sort_values(by='Task_ID', key=sort_func, inplace=True)
            direct_grid_df['idx'] = direct_grid_df.apply(lambda row: int(row['Task_ID'].split('_')[-1]), axis=1)


            object_based_csv_file = os.path.join(object_based_results_path, f'{path}_4.csv')
            object_based_df = pd.read_csv(object_based_csv_file, delimiter=',')
            old_len = len(object_based_df)
            object_based_df = object_based_df.drop_duplicates(subset='Task_ID')
            new_len = len(object_based_df)
            if old_len != new_len:
                print(f'len(object_based_df) != len(direct_grid_df) for {path}', old_len, new_len)
            object_based_df['idx'] = object_based_df.apply(lambda row: int(row['Task_ID'].split('_')[-1]), axis=1)
            object_based_df.sort_values(by='Task_ID', key=sort_func, inplace=True)

            def verify_results(task, task_json_csv):
                for split in ['train', 'test']:
                    for i, t in enumerate(task_json_csv[split]):
                        if t['input'] != task[split][i]['input']:
                            # import ipdb; ipdb.set_trace()
                            raise ValueError('input does not match', ct)
                        if t['output'] != task[split][i]['output']:
                            # import ipdb; ipdb.set_trace()
                            raise ValueError('output does not match', ct)

            for f in files:
                if f.endswith('.json'):
                    json_path = os.path.join(path_this, f)
                    with open(json_path, 'r') as f_:
                        task = json.load(f_)

                    paper_results = {
                        'direct_grid_correct': None,
                        'object_based_correct': None,
                        'file_json': f,
                    }

                    try:
                        # verify that the data matches
                        file_idx = int(f.replace('.json', '').split('_')[-1])
                        result = direct_grid_df[direct_grid_df['idx'] == file_idx]
                        if result.empty or len(result) > 1:
                            # import ipdb; ipdb.set_trace()
                            raise ValueError(f'result is empty for json_file:{f}, file_idx:{file_idx}')
                        else:
                            row = result.iloc[0]
                        verify_results(task, json.loads(row['Task_json']))
                        paper_results['direct_grid_correct'] = row['Match_flag'] == 1
                        result = object_based_df[object_based_df['idx'] == file_idx]
                        if result.empty or len(result) > 1:
                            # import ipdb; ipdb.set_trace()
                            raise ValueError(f'result is empty for json_file:{f}, file_idx:{file_idx}')
                        else:
                            row = result.iloc[0]
                        verify_results(task, json.loads(row['Task_json']))
                        paper_results['object_based_correct'] = row['Match_flag'] == 1

                    except Exception as e:
                        print('Error:', ct, type(e), e)
                        
                    task['task_info'] = {
                        'task_id': ct,
                        'concept': concept,
                        'filename': f,
                        'results': paper_results,
                    }
                    task['descriptions'] = {
                        'description_input': '',
                        'description_output': '',
                        'description_output_grid_size': '',
                    }
                    tasks.append(task)
                    ct_this += 1
                    if ct_this <= 2:
                        sampled_idxs.append(ct)

                    ct += 1

        with open(cache_path, 'wb') as f:
            pickle.dump(tasks, f)

        print('sampled_idxs', len(sampled_idxs), sampled_idxs)
        return tasks

if __name__ == '__main__':
    dataset = get_larc_dataset()
    import ipdb; ipdb.set_trace()
