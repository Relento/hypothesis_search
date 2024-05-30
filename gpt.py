import openai
import os
import tiktoken
import time
import json
from config import cfg
from pprint import pprint
from collections import defaultdict

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        if model == 'gpt4-0613':
            model = 'gpt-4-0314'
        if model == 'gpt35-0613' or model == 'gpt35-0613-16k':
            model = 'gpt-3.5-turbo-0301'
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"Warning: model {model} not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo" or model == "gpt35-0613":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314" or model == "gpt4-0613":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def num_tokens_from_text(text, model="gpt-3.5-turbo-0301"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"Warning: model {model} not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def get_model_price(model_name):
    if model_name in ['gpt-3.5-turbo', 'gpt-3.5-turbo-0301', 'gpt35-0613', 'gpt35-0613-16k']:
        prompt_price = 0.002
        completion_price = 0.002
    elif model_name in ['gpt-4', 'gpt-4-0314', 'gpt4-0613']:
        prompt_price = 0.03
        completion_price = 0.06
    else:
        raise NotImplementedError(f"estimate_price() is not implemented for model {model_name}.")
    
    return prompt_price, completion_price

def estimate_price(model_name, num_tokens, num_completions, max_tokens):
    prompt_price, completion_price = get_model_price(model_name)

    prompt_cost = (prompt_price * num_tokens) / 1000
    completion_cost = (completion_price * num_completions * max_tokens) / 1000
    total_cost = prompt_cost + completion_cost
    return {
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "total_cost": total_cost
    }


class GPTStatsGroup():
    # stats for a specific group
    def __init__(self, name):
        self.name = name
        self.prompt_cost = 0
        self.completion_cost = 0
        # computing the actual number of tokesn completed
        self.completion_cost_real = 0
        self.completed_tokens = 0
        self.num_calls = 0

    
class GPT():
    def __init__(self, openai_key, cache_file="resources/cache.json"):
        self.cache_file = cache_file
        self.exponential_backoff = 1
        # Load the cache JSON file, if cache file exists. Else, cache is {}
        if os.path.exists(cache_file):
            while os.path.exists(self.cache_file + ".tmp") or os.path.exists(self.cache_file + ".lock"):
                time.sleep(0.1)
            with open(cache_file, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

        if cfg.use_azure_api:
            openai.api_type = cfg.azure.openai_api_type
            print("Using Azure API", openai.api_type, "version", cfg.azure.openai_api_version, "base", cfg.azure.openai_api_base, "key", cfg.azure.openai_api_key)
            openai.api_version = cfg.azure.openai_api_version
            openai.api_base = cfg.azure.openai_api_base
            openai.api_key = cfg.azure.openai_api_key
        else:
            openai.organization, openai.api_key = openai_key.split(":")

        self.stats_group = {}
        self.last_call_time = {}
        self.tokens_requested = defaultdict(int)
    
    def print_summary(self):
        print('Summary of API calls:')
        for name, g in self.stats_group.items():
            print(f"  {name}:")
            for key in ['prompt_cost', 'completion_cost', 'completion_cost_real', 'num_calls']:
                value = getattr(g, key)
                if isinstance(value, float):
                    value = round(value, 4)
                print(f"    {key}: {value}")

    def generate(self,
        messages_or_prompt, num_completions=8, max_tokens=500, temperature=0.5, presence_penalty=0.0,
        stop=["\ndef"], indented=True, indented_after_first_line=False, require=None, cache_key=None,
        rate_limit_tokens=16000, verbose=False, logit_bias=None, model_name=None,
        dry_run=False, parsing_func=None, stats_group_name=None
    ):
        if isinstance(messages_or_prompt, str):
            prompt = messages_or_prompt
            messages = []
            messages.append({"role": "system", "content": ""})
            messages.append({"role": "user", "content": prompt})
        else:
            messages = messages_or_prompt

        def assert_messages(messages):
            for idx, message in enumerate(messages):
                assert "role" in message and "content" in message
                if idx == 0:
                    assert message["role"] == "system"
                    continue
                if idx % 2 == 1:
                    assert message["role"] == "user"
                else:
                    assert message["role"] == "assistant"
        assert_messages(messages)

        def parse_response_default(text):
            result = []
            if cfg.directly_output_grid:
                import ast
                lines = text.split("\n")
                splits = []
                for idx, line in enumerate(lines):
                    if line.startswith("Test Output"):
                        splits.append(idx)
                if len(splits) == 0:
                    return "No output found."
                splits.append(len(lines))
                for i in range(len(splits) - 1):
                    matrix_str = "\n".join(lines[splits[i] + 1:splits[i+1]]).replace('[ ', '[').replace('\n', '').replace(' ', ',').replace(',,', ',')
                    try:
                        result.append(ast.literal_eval(matrix_str))
                    except:
                        result.append([[-1]])
                return result

            # np.ndarray[int] needs python 3.9
            text = text.replace('np.ndarray[int]', 'np.ndarray')
            lines = text.split("\n")
            # Sometimes the model will output tags for code snippet that should be removed.
            if cfg.parse_python_tag:
                if cfg.generate_detailed_plan_and_tests:
                    # parse all python blocks
                    in_python_block = False
                    n = len(lines)
                    idx = 0
                    tags = []
                    start_idx = 0
                    lines_new = []
                    while idx < n:
                        l = lines[idx]
                        if l.startswith('```python'):
                            in_python_block = True
                            start_idx = idx + 1
                        elif l.startswith('```'):
                            in_python_block = False
                            lines_new.extend(lines[start_idx:idx])
                        idx += 1
                
                    lines = lines_new

                else:
                    has_python_tag = False
                    start_idx = 0
                    end_idx = len(lines)
                    for i, l in enumerate(lines):
                        if l.startswith('```python'):
                            has_python_tag = True
                            start_idx = i
                            for j in range(i + 1, len(lines)):
                                if lines[j].startswith('```'):
                                    end_idx = j
                                    break
                            break

                    if has_python_tag:
                        lines = lines[start_idx + 1:end_idx]

            lines = list(filter(lambda x: not x.startswith('```'), lines))
            for line_idx, line in enumerate(lines):
                if (indented or (indented_after_first_line and line_idx > 0)) and line.lstrip() == line and line.strip() != "":
                    break
                if require is not None and line.strip() != "" and require not in line:
                    break
                result += [line]
            return result

        if model_name is None:
            model_name = "gpt-3.5-turbo-0301"
        if verbose:
            print(messages)
            print("-----")

        basic_info =  {
            'model_name': model_name,
            'num_tokens': num_tokens_from_messages(messages, model=model_name),
            'num_completions': num_completions,
            'max_tokens': max_tokens,
        }
        estimated_price_dict = estimate_price(**basic_info)
        if model_name in ['gpt-3.5', 'gpt-3.5-turbo-0301', 'gpt-3.5-turbo', 'gpt35-0613']:
            max_context_length = 4095
        elif model_name in ['gpt35-0613-16k']:
            max_context_length = 16383
        elif model_name in ['gpt-4', 'gpt-4-0314', 'gpt4-0613']:
            max_context_length = 8191
        if dry_run:
            return {
                **basic_info,
                'estimated_price': estimated_price_dict['total_cost'],
                'estimated_price_prompt': estimated_price_dict['prompt_cost'],
                'estimated_price_completion': estimated_price_dict['completion_cost'],
                'exceed_context_length': basic_info['num_tokens'] + max_tokens > max_context_length,
            }
        if basic_info['num_tokens'] + max_tokens > max_context_length:
            return "Exceed context length."
        
        if stats_group_name is None:
            stats_group_name = 'default'
        if stats_group_name not in self.stats_group:
            self.stats_group[stats_group_name] = GPTStatsGroup(stats_group_name)

        gpt_stats_group = self.stats_group[stats_group_name]
        gpt_stats_group.prompt_cost += estimated_price_dict['prompt_cost']
        gpt_stats_group.completion_cost += estimated_price_dict['completion_cost']
        gpt_stats_group.num_calls += 1

        cache_key_base = messages if cache_key is None else cache_key
        cache_key_list = (cache_key_base, max_tokens, temperature, stop)
        if presence_penalty != 0.0:
            cache_key_list = cache_key_list + (presence_penalty,)
        if model_name != "code-davinci-002":
            cache_key_list = cache_key_list + (model_name,)
        cache_key = str(cache_key_list)

        if parsing_func is None:
            parse_response = parse_response_default
        else:
            parse_response = parsing_func

        def compute_completion_cost(rets):
            _, completion_cost = get_model_price(model_name)
            num_tokens = 0
            for ret in rets:
                num_tokens += num_tokens_from_text(ret['raw'])
            return num_tokens * completion_cost / 1000

        if cache_key in self.cache:
            if len(self.cache[cache_key]) < num_completions:
                num_completions -= len(self.cache[cache_key])
                results = self.cache[cache_key]
                for r in results:
                    # Update the cache as the parse function may be updated.
                    r['parsed'] = parse_response(r['raw'])
            else:
                cur_implementations = self.cache[cache_key].copy()
                for c in cur_implementations:
                    # Update the cache as the parse function may be updated.
                    c['parsed'] = parse_response(c['raw'])

                gpt_stats_group.completion_cost_real += compute_completion_cost(cur_implementations[:num_completions])
                return cur_implementations[:num_completions]
        else:
            results = []

        if cfg.only_allow_cache:
            raise NotImplementedError('Currently only allow cache')

        if "gpt-4-0314" in model_name:
            print("WARNING, GPT-4 is very expensive. And gpt-4-0314 will be deprecated by June 14, 2023.")

        print("Calling GPT!")
        total_tokens = num_completions * max_tokens
        completions_per_call = rate_limit_tokens // max_tokens

        recorded_completion_tokens = 0
        while total_tokens > 0:
            num_completions = min(total_tokens // max_tokens, completions_per_call)
            print(num_completions, "completions", max_tokens, "tokens each")
            start_time = time.time()
            while True:
                try:
                    time.sleep(1)
                    if logit_bias is None:
                        response = openai.ChatCompletion.create(
                            model=None if cfg.use_azure_api else model_name,
                            engine=None if not cfg.use_azure_api else model_name,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            presence_penalty=presence_penalty,
                            stop=stop,
                            n=num_completions,
                        )
                        completions = response['choices']
                    else:
                        response = openai.ChatCompletion.create(
                            model=None if cfg.use_azure_api else model_name,
                            engine=None if not cfg.use_azure_api else model_name,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            presence_penalty=presence_penalty,
                            stop=stop,
                            n=num_completions,
                            logit_bias=logit_bias
                        )
                        completions = response['choices']
                    recorded_completion_tokens += response['usage']['completion_tokens']
                    self.exponential_backoff = 1
                    break
                except openai.error.RateLimitError:
                    print("Rate limit reached. Waiting before retrying...")
                    time.sleep(16 * self.exponential_backoff)
                    self.exponential_backoff *= 2
                except openai.error.APIError as e:
                    print(f"OpenAI API returned an API Error: {e}")
                    pass
            for completion in completions:
                results.append({
                    'parsed': parse_response(completion['message']['content']),
                    'raw': completion['message']['content'],
                })

            end_time = time.time()
            print('API Call time elapsed:', end_time - start_time, 'Num tokens:', recorded_completion_tokens)

            # Save updated cache - reopen in case multiple processes running
            # Save to a temp file first, then rename
            # Check if a temp file exists, and if so, wait for it to be deleted
            while os.path.exists(self.cache_file + ".tmp") or os.path.exists(self.cache_file + ".lock"):
                time.sleep(0.1)
            start_time = time.time()
            # create an empty file to indicate that we are writing to the cache
            with open(self.cache_file + ".lock", "w") as f:
                pass
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r") as f:
                    self.cache = json.load(f)
            # do not cache the parsed result
            results_cache = []
            for r in results:
                results_cache.append({
                    'raw': r['raw'],
                })
            self.cache[cache_key] = results_cache

            with open(self.cache_file + ".tmp", "w") as f:
                json.dump(self.cache, f)
            os.rename(self.cache_file + ".tmp", self.cache_file)
            os.remove(self.cache_file + ".lock")
            end_time = time.time()
            print('Dump to json time elapsed:', end_time - start_time)
            total_tokens -= num_completions * max_tokens
        
        gpt_stats_group.completion_cost_real += compute_completion_cost(results)
        return results


if __name__ == '__main__':
    from config import cfg
    gpt = GPT(openai_key=cfg.openai_api_key)
    input_prompt = \
'''
You will be given a list of input output pairs. Each input and output is a grid of numbers (from 0 to 9, int not str). Your job is to infer the python program that transforms the input grid to the corresponding output grid. The input-output pairs are given below:
Example 0:
Input:
[[1 0 0 5 0 1 0]
 [0 1 0 5 1 1 1]
 [1 0 0 5 0 0 0]]
Output:
[[0 0 0]
 [0 2 0]
 [0 0 0]]
Example 1:
Input:
[[1 1 0 5 0 1 0]
 [0 0 1 5 1 1 1]
 [1 1 0 5 0 1 0]]
Output:
[[0 2 0]
 [0 0 2]
 [0 2 0]]
Example 2:
Input:
[[0 0 1 5 0 0 0]
 [1 1 0 5 1 0 1]
 [0 1 1 5 1 0 1]]
Output:
[[0 0 0]
 [2 0 0]
 [0 0 2]]
Now, please write a python program transform_grid(input_grid: List[List[int]]) -> List[List[int]] that transforms the input grid to the corresponding output grid.
Hint: You may want to use the following guidance to implement the function: 
In the input, you should see a grid with a gray line down the center The output grid size should be 3x3 To make the output, you have to  look at both the left and right parts of the input grid. You will notice that the left and right parts are 3x3. For each square that is colored on both the left and right parts, color the output grid with red on the new 3x3.
The number in the input grid can be mapped to the following colors:0:black; 1:blue; 2:red; 3:green; 4:yellow; 5:grey; 6:fuschia; 7:orange; 8:teal; 9:brown
Just reply with the implementation of transform_grid(input_grid: List[List[int]]) and nothing else, each cell in the output should only be numbers from 0 to 9.
'''
    print(gpt.generate(messages_or_prompt=input_prompt, 
                        num_completions=1, 
                        max_tokens=1000, 
                        dry_run=False, 
                        indented=False,
                        indented_after_first_line=True))
    import ipdb; ipdb.set_trace()
