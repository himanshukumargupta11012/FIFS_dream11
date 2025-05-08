#!/bin/bash

# Create folders
mkdir -p fantasy_team_selection/{data/{raw,processed,embeddings,predictions},src/{data_preprocessing,embeddings,models,team_selection,inference},notebooks,configs,scripts,tests}

# Create files in fantasy_team_selection/data
touch fantasy_team_selection/data/constraints.json

# Create files in fantasy_team_selection/src/data_preprocessing
touch fantasy_team_selection/src/data_preprocessing/load_data.py
touch fantasy_team_selection/src/data_preprocessing/feature_engineering.py
touch fantasy_team_selection/src/data_preprocessing/dataset.py
touch fantasy_team_selection/src/data_preprocessing/utils.py

# Create files in fantasy_team_selection/src/embeddings
touch fantasy_team_selection/src/embeddings/player_embedding.py
touch fantasy_team_selection/src/embeddings/venue_embedding.py
touch fantasy_team_selection/src/embeddings/opponent_embedding.py

# Create files in fantasy_team_selection/src/models
touch fantasy_team_selection/src/models/fantasy_point_model.py
touch fantasy_team_selection/src/models/train.py
touch fantasy_team_selection/src/models/evaluate.py
touch fantasy_team_selection/src/models/config.yaml

# Create files in fantasy_team_selection/src/team_selection
touch fantasy_team_selection/src/team_selection/team_builder.py
touch fantasy_team_selection/src/team_selection/captain_vc_selector.py
touch fantasy_team_selection/src/team_selection/constraints_checker.py

# Create files in fantasy_team_selection/src/inference
touch fantasy_team_selection/src/inference/generate_predictions.py
touch fantasy_team_selection/src/inference/select_team.py
touch fantasy_team_selection/src/inference/visualize.py

# Create files in fantasy_team_selection/notebooks
touch fantasy_team_selection/notebooks/data_exploration.ipynb
touch fantasy_team_selection/notebooks/model_training.ipynb
touch fantasy_team_selection/notebooks/team_selection.ipynb

# Create files in fantasy_team_selection/configs
touch fantasy_team_selection/configs/config.yaml
touch fantasy_team_selection/configs/model_params.yaml
touch fantasy_team_selection/configs/team_selection.json
touch fantasy_team_selection/configs/environment.yaml

# Create files in fantasy_team_selection/scripts
touch fantasy_team_selection/scripts/train_model.sh
touch fantasy_team_selection/scripts/run_inference.sh
touch fantasy_team_selection/scripts/preprocess_data.sh
touch fantasy_team_selection/scripts/visualize_results.sh

# Create files in fantasy_team_selection/tests
touch fantasy_team_selection/tests/test_data_pipeline.py
touch fantasy_team_selection/tests/test_models.py
touch fantasy_team_selection/tests/test_team_selection.py
touch fantasy_team_selection/tests/test_end_to_end.py

# Create files in fantasy_team_selection root
touch fantasy_team_selection/requirements.txt
touch fantasy_team_selection/environment.yml
touch fantasy_team_selection/README.md
touch fantasy_team_selection/.gitignore
touch fantasy_team_selection/LICENSE

echo "Folder structure created successfully."
