num_threads=16

python data_processing/data_update.py -m recently_added_30_male -o data/raw/cricsheet -t $num_threads
python data_processing/data_processing.py --input_dir data/raw/cricsheet/ --output_dir data/interim/ --num_threads $num_threads
python data_processing/feature_engineering.py --input data/interim/ODI_all.csv --output_dir data/processed --window 7 --threads $num_threads --output_file final
python model/mlp_baseline_final.py -f 7_final -e 20 -dim 128 -batch_size 1024 -lr 0.005 -model_name test
python model/predict_model.py data/processed/match1_squad.csv