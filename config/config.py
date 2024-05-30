from . import yacs
from .yacs import CfgNode as CN
import argparse
import os
import time

cfg = CN()

# experiment name
cfg.exp_name = 'hello'
# result
cfg.results_dir = 'results'

cfg.openai_api_key = ''
cfg.larc_dataset_path = ''
cfg.larc_cache_path = ''
cfg.only_load_sucess_task = False

cfg.num_completions = 1
cfg.max_tokens = 100
cfg.add_hint = False
cfg.model_name = 'gpt-3.5-turbo-0301'

def parse_cfg(cfg, args):
    time_id = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    # assign the gpus
    cfg.results_dir = os.path.join(cfg.results_dir, cfg.exp_name, cfg.model_name, time_id)
    cfg.cfg_file = args.cfg_file

def make_cfg(args):
    with open(args.cfg_file, 'r') as f:
        current_cfg = yacs.load_cfg(f)

    if 'parent_cfg' in current_cfg.keys():
        with open(current_cfg.parent_cfg, 'r') as f:
            parent_cfg = yacs.load_cfg(f)
        cfg.merge_from_other_cfg(parent_cfg)

    cfg.merge_from_other_cfg(current_cfg)
    cfg.merge_from_list(args.opts)

    parse_cfg(cfg, args)
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="resources/configs/default.yaml", type=str)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
cfg = make_cfg(args)