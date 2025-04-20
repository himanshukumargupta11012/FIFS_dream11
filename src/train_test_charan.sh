num_threads=16
window=7

python data_processing/load_data_charan.py -m ipl -o data_charan/raw/cricsheet -t $num_threads
python data_processing/data_processing_charan.py --input_dir data_charan/raw/cricsheet/ --output_dir data_charan/interim/ --num_threads $num_threads
# python data_processing/feature_engineering_charan.py --input data_charan/interim/T20_all.csv --output_dir data_charan/processed --window $window --threads $num_threads --output_file IPL
# python model/lgbm_final_charan.py --data_file data/processed/combined/7_IPL.csv
# python model/mlp_final_charan.py --data_file data/processed/combined/7_IPL.csv --epochs 57
python model/predict_model.py --input_path testing_data/SquadPlayerNames_IndianT20League\ -\ Match_38.csv --ensemble