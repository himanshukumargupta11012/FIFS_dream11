num_threads=16
window=7

python data_processing/load_data_charan.py -m ipl -o data_charan/raw/cricsheet -t $num_threads
python data_processing/data_processing_charan.py --input_dir data_charan/raw/cricsheet/ --output_dir data_charan/interim/ --num_threads $num_threads
# python data_processing/feature_engineering_charan.py --input data_charan/interim/T20_all.csv --output_dir data_charan/processed --window $window --threads $num_threads --output_file IPL
# python model/mlp_baseline_final.py -f "$window"_IPL -e 20 -dim 128 -batch_size 1024 -lr 0.005 -model_name test
python predict_model.py --input_path /home/himanshu/D11_midprep_FIFS/src/data/processed/SquadPlayerNames_IndianT20League - Match_36.csv