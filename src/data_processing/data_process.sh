#!/bin/bash

# ./data_process.sh ../data/raw/cricksheet/ ../data/interim/ ../data/processed/ 15 <num_threads>
# choose number of threads based on compute available 

# if [[ $# -ne 5 ]]; then
#     echo "Usage: $0 <input_dir> <output_dir1> <output_dir2> <window> <num_threads>"
#     exit 1
# fi

# if [[ ! -d "$input_dir" ]]; then
#     echo "Error: Input directory '$input_dir' does not exist."
#     exit 1
# fi

cricksheet_data_dir="$1"
output_dir1="$2"
output_dir2="$3"
window="$4"
NUM_THREADS="$5"

# Ensure necessary output directories exist
mkdir -p "$cricksheet_data_dir"
mkdir -p "$output_dir1"
mkdir -p "$output_dir2"

export output_dir1
export output_dir2
export window
export NUM_THREADS


# Measure total script start time
total_start_time=$(date +%s)

# Step 1: Download and extract data
echo "Starting Step 1: Downloading and extracting data"
step1_start_time=$(date +%s)
python data_download.py $cricksheet_data_dir $NUM_THREADS
step1_end_time=$(date +%s)
echo "Step 1 completed in $((step1_end_time - step1_start_time)) seconds."

# Step 2: Process files in parallel using data_processing.py
echo "Starting Step 2: Data Processing"
step2_start_time=$(date +%s)
python3 data_processing.py $cricksheet_data_dir $output_dir1 $NUM_THREADS
step2_end_time=$(date +%s)
echo "Step 2 completed in $((step2_end_time - step2_start_time)) seconds."

# Step 3: Parallel feature engineering for each file in output_dir1
echo "Starting Step 3: Feature Engineering"
step3_start_time=$(date +%s)
feature_engineering() {
    local file_path="$1"
    local file_name
    file_name=$(basename "$file_path")
    python3 feature_engineering.py --input "$file_path" --output_dir "$output_dir2" --window "$window" --threads "$NUM_THREADS"
}
export -f feature_engineering
find "$output_dir1" -type f ! -name 'matches_info.csv' | parallel -j "$NUM_THREADS" feature_engineering {}
step3_end_time=$(date +%s)
echo "Step 3 completed in $((step3_end_time - step3_start_time)) seconds."


# Measure total script end time
total_end_time=$(date +%s)
echo "Total script execution time: $((total_end_time - total_start_time)) seconds."