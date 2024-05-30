export SKIP_PRESS_Y=1
python parc.py add_hint False  num_completions 80 \
 dataset arc1d \
 visualize False \
 split_path resources/splits/arc1d_18x6.json \
 model_name gpt4-0613 num_feedbacks 0 \
 language_generation.model_name gpt4-0613 \
 temperature 0.7 \
 grid_array_type numpy \
 parse_python_tag True