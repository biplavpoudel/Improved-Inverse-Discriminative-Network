import requests
import rarfile
import os
from tqdm import tqdm
import time
cedar_url = r'https://www.cedar.buffalo.edu/NIJ/data/signatures.rar'
filename = 'cedar.rar'
extract_path = r'D:/MLProjects/Inverse-Discriminative-Network/dataset/CEDAR'


def download_zip():
    if os.path.exists(filename):
        print("Zip File already exists!")
        return True
    else:
        print("Downloading zip file...")
        r = requests.get(url=cedar_url, stream=True)
        r.raise_for_status()
        if r.status_code == 200:
            total_size = int(r.headers.get('content-length', 0))
            # initializing tqdm progress bar
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
            with open(filename, 'wb') as f:
                for chunks in r.iter_content(chunk_size=1024):
                    if chunks:
                        f.write(chunks)
                        progress_bar.update(len(chunks))
            progress_bar.close()
        print("Zip Downloaded successfully!")
        return True


def extract_rar():
    try:
        init_time = time.time()
        print("Extraction Initiated")
        rarfile.RarFile(filename).extractall(path=extract_path)
    except Exception as e:
        print(f"Extraction failed: {e}")
    else:
        complete_time = time.time()
        print(f"Extraction successful: {filename} to {extract_path} with time: {complete_time - init_time}")


if __name__ == '__main__':
    if download_zip():
        extract_rar()
