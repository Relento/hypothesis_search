export SKIP_PRESS_Y=1
python parc.py add_hint False  num_completions 1 \
 visualize False \
 split_path resources/splits/random_100.json \
 model_name gpt4-0613 num_feedbacks 0 directly_output_grid True add_space_first_number False \
 directly_output_grid_vote False \
 description_select_by_heuristic True \
 azure.openai_api_base https://parc-swiss.openai.azure.com/ \
 azure.openai_api_key 3fd6a43bcfdd4cbf9ab4dfa42cc12a1b \
 temperature 0.5