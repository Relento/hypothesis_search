export SKIP_PRESS_Y=1
python parc.py add_hint True  num_completions 8 \
 use_azure_api True \
 visualize False \
 split_path resources/splits/random_100.json \
 model_name gpt4-0613 num_feedbacks 3 \
 language_generation.model_name gpt4-0613 \
 temperature 0.7 \
 parse_python_tag True \
 use_generated_language True \
 language_generation.manually_select_file resources/language_selection/gpt4_0613_100.json \
 language_generation.add_meta_hint True \
 grid_array_type numpy \
 hint_may_be_wrong True \
 language_generation.visualize_every_solution False \
 load_tasks [0,20,31,37,46,48,51,56,64,72,74,77,97,99,107,115,126,128,148,151,159,170,185,187,210,221,228,247,249,250,255,257,265,289,293,297,316,320,321,324,336,346,351,355,359,370,385,390,397] \
 language_generation.num_completions 64 \
 language_generation.temperature 1.0 \
 language_generation.example_task_ids [139,235] \
 only_last_feedback True