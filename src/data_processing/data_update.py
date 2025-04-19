# python data_update.py -m odis -o ../data/raw/cricsheet -t 16

from concurrent.futures import ThreadPoolExecutor
import requests
import os
import zipfile
import pandas as pd
import shutil
import argparse


def download_chunk(start_byte, end_byte, url, output_file):
    headers = {'Range': f"bytes={start_byte}-{end_byte}"}
    response = requests.get(url, headers=headers, stream=True)
    with open(output_file, 'r+b') as file:
        file.seek(start_byte)
        file.write(response.content)

def create_readme_df(readme_path):
    with open(readme_path, "r") as file:
        match_data = [line[:-1].split(" - ") for line in file if line[:4].isdigit()]

    df = pd.DataFrame(match_data, columns=["date", "level", "type", "gender", "match_id", "teams"])
    df['date'] = pd.to_datetime(df['date'])
    df['match_id'] = df['match_id'].astype(int)    

    return df

# Function to download the file in parallel
def download_file(url, output_filename):

    # try:
        # Get file size from the server
        response = requests.head(url)
        file_size = int(response.headers.get('Content-Length', 0))
        
        # Set up the file to store the download
        with open(output_filename, 'wb') as f:
            f.truncate(file_size)

        # Number of threads to split the download into
        chunk_size = file_size // num_threads

        # Create a thread pool to download chunks in parallel
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i in range(num_threads):
                start_byte = i * chunk_size
                # Make sure the last chunk goes up to the file size
                end_byte = start_byte + chunk_size - 1 if i < num_threads - 1 else file_size - 1
                futures.append(executor.submit(download_chunk, start_byte, end_byte, url, output_filename))

            # Wait for all threads to finish
            for future in futures:
                future.result()

    # except (requests.RequestException, TimeoutError) as e:
    #     print(f"Error during download: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download data from cricsheet.org')
    parser.add_argument('--match_type', '-m', type=str, help='Type of match to download', default='all_json')
    parser.add_argument('--output_dir', '-o', type=str, help='Output directory to store the data', required=True)
    parser.add_argument('--num_threads', '-t', type=int, help='Number of threads to use for downloading', default=4)
    parser.add_argument('--afghanistan_matches', '-a', action='store_true', help='Download Afghanistan matches')
    parser.add_argument('--new', '-n', action='store_true', help='Delete old data and download new data')

    args = parser.parse_args()

    match_type = args.match_type
    output_dir = args.output_dir
    num_threads = args.num_threads

    os.makedirs(output_dir, exist_ok=True)


    data_url = f"https://cricsheet.org/downloads/{match_type}_json.zip"
    people_url = "https://cricsheet.org/register/people.csv"
    name_url = "https://cricsheet.org/register/names.csv"

    if args.afghanistan_matches:
        data_url = "https://web.archive.org/web/20240913035347/" + data_url


    input_urls = [data_url, people_url, name_url]

    output_file_names = [url.split("/")[-1] for url in input_urls]

    
    if not args.afghanistan_matches:
        for idx, url in enumerate(input_urls):
            download_file(url, os.path.join(output_dir, output_file_names[idx]))

    temp_json_dir = os.path.join(output_dir, "temp")
    json_dir = os.path.join(output_dir, "all_json")
    prev_readme_path = os.path.join(output_dir, "matches_info.csv")
    match_type = "all"
    match_gender = "male"

    # Remove old data if --new flag is set
    if args.new:
        shutil.rmtree(json_dir, ignore_errors=True)
        if os.path.exists(prev_readme_path):
            os.remove(prev_readme_path, ignore_errors=True)

    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(temp_json_dir, exist_ok=True)

    # Unzip the downloaded file
    with zipfile.ZipFile(os.path.join(output_dir, output_file_names[0]), 'r') as zip_ref:
        zip_ref.extractall(temp_json_dir)


    # Extract the match information
    readme_path = os.path.join(temp_json_dir, "README.txt")
    readme_df = create_readme_df(readme_path)
    print(f"Number of matches in README: {readme_df.shape[0]}")
    specific_matches = readme_df[((readme_df['type'] == match_type) | (match_type == "all")) & ((readme_df["gender"] == match_gender) | (match_gender == "all")) & (readme_df['match_id'] != 1229824)]
    specific_matches["date"] = specific_matches['date'].dt.strftime("%Y-%m-%d")


    if os.path.exists(prev_readme_path):
        prev_readme_df = pd.read_csv(prev_readme_path)
        not_present_matches = specific_matches[~specific_matches["match_id"].isin(prev_readme_df["match_id"])]    
        final_readme_df = pd.concat([prev_readme_df, not_present_matches])
    else:
        not_present_matches = specific_matches
        final_readme_df = not_present_matches

    print(f"Number of new matches: {not_present_matches.shape[0]}\nTotal matches: {final_readme_df.shape[0]}")

    for match_id in not_present_matches["match_id"]:
        match_file = os.path.join(temp_json_dir, f"{match_id}.json")
        shutil.move(match_file, json_dir)

    final_readme_df.to_csv(prev_readme_path, index=False)

    # remove the temp directory and downloaded zip file
    shutil.rmtree(temp_json_dir)
    if not args.afghanistan_matches:
        os.remove(os.path.join(output_dir, output_file_names[0]))