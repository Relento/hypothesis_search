export SKIP_PRESS_Y=1
python parc.py add_hint True  num_completions 4 \
 dataset arc1d \
 visualize False \
 split_path resources/splits/arc1d_18x6.json \
 model_name gpt4-0613 num_feedbacks 0 \
 language_generation.model_name gpt4-0613 \
 temperature 0.7 \
 use_generated_language True \
 language_generation.add_meta_hint True \
 grid_array_type numpy \
 hint_may_be_wrong True \
 language_generation.visualize_every_solution False \
 language_generation.example_task_ids [139,235] \
 language_generation.num_completions 16 \
 language_generation.temperature 1.0 \
 only_last_feedback True \
 parse_python_tag True