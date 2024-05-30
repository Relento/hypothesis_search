export SKIP_PRESS_Y=1
python parc.py add_hint False num_completions 64 \
 use_azure_api True \
 visualize False \
 split_path resources/splits/random_100.json \
 model_name gpt4-0613 num_feedbacks 0 \
 temperature 0.7 \
 grid_array_type numpy \
 parse_python_tag True \
 description_select_by_heuristic True \
 add_meta_hint True