from concurrent.futures import ThreadPoolExecutor
import requests
import os
import sys
import zipfile
import pandas as pd
import shutil

# Check if the correct number of arguments are passed
if len(sys.argv) != 4:
    print("Usage: python3 data_download.py <match type> <output_dir> <num_threads>")
    sys.exit(1)  # Exit the script with an error code

_, match_type, output_dir, num_threads = sys.argv
num_threads = int(num_threads)

data_url = f"https://cricsheet.org/downloads/{match_type}_json.zip"
people_url = "https://cricsheet.org/register/people.csv"
name_url = "https://cricsheet.org/register/names.csv"

input_urls = [data_url, people_url, name_url]

output_file_names = [url.split("/")[-1] for url in input_urls]

def download_chunk(start_byte, end_byte, url, output_file):
    headers = {'Range': f"bytes={start_byte}-{end_byte}"}
    response = requests.get(url, headers=headers, stream=True)
    with open(output_file, 'r+b') as file:
        file.seek(start_byte)
        file.write(response.content)

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


for idx, url in enumerate(input_urls):
    download_file(url, os.path.join(output_dir, output_file_names[idx]))

# Unzip the data file
json_dir = os.path.join(output_dir, "all_json")
os.makedirs(json_dir, exist_ok=True)

with zipfile.ZipFile(os.path.join(output_dir, output_file_names[0]), 'r') as zip_ref:
    zip_ref.extractall(json_dir)


# Extract the match information
readme_path = os.path.join(json_dir, "README.txt")
with open(readme_path, "r") as file:
    match_data = [line[:-1].split(" - ") for line in file if line[:4].isdigit()]

df = pd.DataFrame(match_data, columns=["date", "type", "format", "gender", "match_id", "teams"])
df['date'] = pd.to_datetime(df['date'])
df.to_csv(f"{output_dir}/matches_info.csv", index=False)

shutil.move(readme_path, output_dir)