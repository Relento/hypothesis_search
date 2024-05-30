import json
import re
import tiktoken
import re
from config import cfg
import openai
import time

if cfg.use_azure_api:
    openai.api_type = cfg.azure.openai_api_type
    print("Using Azure API", openai.api_type, "version", cfg.azure.openai_api_version, "base", cfg.azure.openai_api_base, "key", cfg.azure.openai_api_key)
    openai.api_version = cfg.azure.openai_api_version
    openai.api_base = cfg.azure.openai_api_base
    openai.api_key = cfg.azure.openai_api_key

cache_path = "resources/summarization_cache.json"

def call_chatgpt(messages, model="gpt-3.5-turbo", **kwargs):
    # Load cache from file
    try:
        with open(cache_path, "r") as cache_file:
            cache = json.load(cache_file)
    except FileNotFoundError:
        cache = {}
    all_data = str((
        str(messages),
        str(model),
        str(kwargs)
    ))
    message_hash = str(all_data)
    if message_hash in cache:
        return cache[message_hash]
    
    raise NotImplementedError("Only allowed cache for now.")

    backoff = 1

    while True:
        try:
            time.sleep(backoff)
            response = openai.ChatCompletion.create(
                model=None if cfg.use_azure_api else model,
                engine=None if not cfg.use_azure_api else model,
                messages=messages,
                **kwargs
            )
            break
        except openai.error.RateLimitError as e:
            print("Rate limit error, backing off")
            backoff = (backoff + 1) * 2

    result = [choice.message.content for choice in response.choices]
    if len(result) == 1:
        result = result[0]
    cache[message_hash] = result
    # Save cache to file after processing
    with open(cache_path, "w") as cache_file:
        json.dump(cache, cache_file)
    return result

def extract_items(numbered_list_str):
    # Use regular expression to match lines that start with a number followed by any number of digits and periods
    # and then capture the text that follows (excluding any leading/trailing whitespace).
    pattern = re.compile(r'^\s*\d+(\.\d+)*[\.\)]\s*(.*?)\s*$', re.MULTILINE)
    matches = pattern.findall(numbered_list_str)
    # Extract the second element of each tuple to get the item text
    items = [match[-1] for match in matches]
    return items

def num_tokens_from_messages(messages, model="gpt-4"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
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
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


# Function to read the JSON file and return the data
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def print_task(task_id_to_description_outputs, task_id, n=64):
    print(f'Task ID: {task_id}')
    print('Description outputs:')
    for i, description_output in enumerate(task_id_to_description_outputs[task_id][:n]):
        print(f"{i + 1}. {description_output}")
    print('')

# Function to create a dictionary mapping task_id to a list of description_outputs
def create_task_id_to_description_outputs_map(data, key_type="parsed"):
    task_id_to_description_outputs = {}
    for item in data:
        task_id = item.get('task_id')
        results = item.get('results', [])
        description_outputs = []
        for result_idx, result in enumerate(results):
            if result.get(key_type):
                try:
                    description_outputs.append(
                        (
                            result_idx,
                            result.get(key_type).get('description_output')
                        )
                    )
                except Exception as e:
                    description_outputs.append(
                        (
                            result_idx,
                            None
                        )
                    )
                    pass
        if task_id is not None and description_outputs:
            task_id_to_description_outputs[task_id] = description_outputs
    return task_id_to_description_outputs

def make_message(chat_gpt_prompt, system="You are a genius solving language puzzles."):
    messages = [
        {
            'role': 'system',
            'content': system
        },
        {
            'role': 'user',
            'content': chat_gpt_prompt
        }
    ]
    return messages


# Main script
file_path = 'resources/generated_languages/gpt4_0613_100_64.json'
data = read_json_file(file_path)
task_id_to_description_outputs = create_task_id_to_description_outputs_map(data)
encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')

# Load from synthesized_language.json if it exists
try:
    with open('resources/generated_languages/gpt4_0613_100_synthesized_raw.json', 'r') as f:
        synthesized_language = json.load(f)
        # Replace str keys with int keys
        synthesized_language = {int(k): v for k, v in synthesized_language.items()}
except:
    synthesized_language = {}

for task_id, languages in task_id_to_description_outputs.items():
# for task_id in task_ids_to_check:
    if task_id in synthesized_language:
        continue
    languages = task_id_to_description_outputs[task_id]
    base_chat_gpt_prompt = (
"""Given a list of rules, categorize them into eight distinct categories based on their similarities. For each category, synthesize the rules into a single, specific rule that combines the ideas of all rules in that category, while clearly differentiating it from the other categories.
The new rule should be as specific as possible, following the format of the given rules.
The new rule should be applicable without any information from the original rules - i.e. it should be standalone.

Rules:
"""
    )
    batch_size = 64
    num_requested = 1000
    max_tokens = 8192
    cur_batch_idx = 0
    chat_gpt_prompt = base_chat_gpt_prompt[::]
    correspondance = {}
    print(f"Task ID: {task_id}")
    for language_idx, language in languages:
        if language is None:
            print(f"Skipping {language_idx} because it is None")
            continue
        print_index = (cur_batch_idx % batch_size) + 1
        correspondance[print_index] = language_idx
        chat_gpt_prompt += f"{print_index}. {language}\n"
        cur_batch_idx += 1
        if cur_batch_idx % batch_size == 0 or language_idx == languages[-1][0]:
            messages = make_message(chat_gpt_prompt)
            num_tokens = num_tokens_from_messages(messages) + num_requested
            print(f"Num tokens: {num_tokens}")
            if num_tokens >= max_tokens:
                overshot = num_tokens - max_tokens + 1
                print(f"Overshot by {overshot} tokens")
                cur_num_requested = num_requested - overshot
                print(f"Reducing num_requested to {cur_num_requested}")
            else:
                cur_num_requested = num_requested
            response = call_chatgpt(messages, temperature=0.0, max_tokens=cur_num_requested, model='gpt4-0613')
            
            new_message = make_message(
                """Given a list of categories, each of which has one general rule, extract the rule from each category, verbatim. Create a numbered list of rules, one per category, in the format "{number}. {rule}" where {number} is the number of the category and {rule} is the rule in that category. "If there happen to be multiple rules in a category, just extract one of them."
                """ + response,
                system="You are an expert natural language parser. Return just the transformed language.",
            )
            # Extract all standalone numbers from the response
            new_response = call_chatgpt(new_message, temperature=0.0, max_tokens=cur_num_requested, model='gpt35-0613')
            extracted_list = extract_items(new_response)
            synthesized_language[task_id] = {}
            synthesized_language[task_id]['extracted_list'] = extracted_list
            synthesized_language[task_id]['original_response'] = response
            synthesized_language[task_id]['new_response'] = new_response
            chat_gpt_prompt = base_chat_gpt_prompt
            # # Save the matched indices to a JSON file
            with open('resources/generated_languages/gpt4_0613_100_synthesized_raw.json', 'w') as file:
                json.dump(synthesized_language, file, indent=2)
            

# a cleaner version for loading
for task_id in synthesized_language:
    synthesized_language[task_id] = synthesized_language[task_id]['extracted_list']

with open('resources/generated_languages/gpt4_0613_100_synthesized.json', 'w') as file:
    json.dump(synthesized_language, file, indent=2)

                

            
            