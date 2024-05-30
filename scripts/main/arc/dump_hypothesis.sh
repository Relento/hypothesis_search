export SKIP_PRESS_Y=1
python language_generation.py split_path resources/splits/random_100.json \
    language_generation.example_task_ids [139,235] model_name gpt4-0613 \
    language_generation.model_name gpt4-0613 \
    language_generation.add_meta_hint True \
    language_generation.num_completions 64 \
    language_generation.temperature 1.0