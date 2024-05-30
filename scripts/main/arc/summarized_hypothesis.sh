export SKIP_PRESS_Y=1
python parc.py add_hint True  num_completions 8 \
 visualize True \
 split_path resources/splits/random_100.json \
 model_name gpt4-0613 num_feedbacks 3\
 temperature 0.7 \
 use_generated_language True \
 language_generation.add_meta_hint True \
 grid_array_type numpy \
 language_generation.visualize_every_solution False \
 language_generation.example_task_ids [139,235] \
 language_generation.num_completions 0 \
 language_generation.temperature 1.0 \
 language_generation.read_from_file resources/generated_languages/gpt4_0613_100_synthesized.json \
 program_clustering True \
 parse_python_tag True