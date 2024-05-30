export SKIP_PRESS_Y=1
python parc.py add_hint True  num_completions 8 \
 visualize False \
 split_path resources/splits/random_100.json \
 model_name gpt4-0613 num_feedbacks 3 \
 temperature 0.7 \
 grid_array_type numpy \
 description_select_by_heuristic True \
 parse_python_tag True