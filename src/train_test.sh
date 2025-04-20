num_threads=16
window=7

python data_processing/data_update.py -m ipl -o data/raw/cricsheet -t $num_threads
python data_processing/data_processing.py --input_dir data/raw/cricsheet/ --output_dir data/interim/ipl --num_threads $num_threads
# python data_processing/feature_engineering.py --input_dir data/interim/ipl --output_dir data/processed --window $window --threads $num_threads --output_file IPL
# python model/transformer_himanshu.py -f "$window"_IPL -e 20 -dim 128 -batch_size 1024 -lr 0.005 -model_name test
python model/predict_model.py --input_path testing_data/SquadPlayerNames_IndianT20League\ -\ Match_38.csv --model_name transformer_final_d-7_IPL_sd-2010-01-01_ed-2025-05-01