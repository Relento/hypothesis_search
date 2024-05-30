# Hypothesis Search: Inductive Reasoning with Language Models

This is the implementation of the paper **Hypothesis Search: Inductive Reasoning with Language Models**. 
[[Paper]](https://arxiv.org/pdf/2309.05660)

# Installation
```
conda create -n hypothesis_search python=3.8.16
conda activate hypothesis_search
pip install -r requirements.txt
# unzip the prompt cache file
unzip resources/cache.json.zip
```
The code is tested on Ubuntu 20.04.

**Important**: We have cached all our results in `resources/cache.json`. So all experiments can be run without having an OpenAI API key. In our pipeline, error messages of Python program execution will be included in the prompt during the feedback stage. To leverage the cache to reproduce all results, please update line 250 of `utils.py` so the path strings in error message are mapped to environment-independent strings.
```
...
current_root = os.getcwd()
path_mapping = {
    # mapping machine-specific paths to generic paths
    current_root: '/root',
    '/data2/rcwang/miniconda3/': '/envs',
    # add your own python package mapping here
    # an example:
    '/home/rcwang/miniconda3/envs/hypothesis_search/': '/envs',
...
```
# Running experiments

## Main Results
Main results can be obtained by running the scripts in `scripts/main`:
```
scripts/
└── main
    ├── 1darc
    │   ├── full.sh
    │   └── program_only.sh
    └── arc
        ├── direct.sh
        ├── human_selected_hypothesis.sh
        ├── human_written_hypothesis.sh
        ├── program_only.sh
        └── summarized_hypothesis.sh
```
To get results with different numbers of feedbacks, alter the `num_feedbacks $NUMBER` argument in the script. Currently we only include results on the ARC and 1D-ARC datasets.
## Results Visualization
For each experiment, you can turn on the flag `visualize True` to visualize the generated hypotheses, programs and execution results as a webpage. The webpage will be saved in the `results` directory. We provide an example in `results/naive/gpt4-0613/summarized_hypothesis/vis/index.html`.

## 100 Randomly Sampled Tasks of ARC
The task IDs of the 100 randomly sampled ARC tasks used in our experiments are stored in `resources/splits/random_100.json`. We use the task IDs in [LARC](https://samacquaviva.com/LARC/). The corresponding JSON file name of each task in the original ARC dataset is stored in `resources/id2json.json`.

## Hypothesis Generation / Summarization
For the selected 100 tasks, we can dump the generated hypotheses by running `sh scripts/main/arc/dump_hypothesis.sh`; 64 hypotheses for each task will be saved in `resources/generated_languages/gpt4_0613_100_64.json`. Run `python language_summarization.py` to summarize the 64 hypotheses to 8 hypotheses; results will be saved in `resources/generated_languages/gpt4_0613_100_synthesized.json`. 

## Running New Tasks
Currently the code can only run examples in the cache by default. OpenAI API calls are disabled to prevent unnecessary cost. To actually use OpenAI API calls to generate new results, put your API key in the `openai_api_key` field in `resources/configs/default.yaml` and set `only_allow_cache` to `false`. It's strongly encouraged to set `num_feedbacks $NUMBER` to be <= 1 to reduce cost.