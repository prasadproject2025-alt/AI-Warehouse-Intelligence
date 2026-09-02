import os
import requests
import json
import time

with open('data_videos.json') as f:
    videos = json.load(f)

os.makedirs('data/raw', exist_ok=True)

def download_file_from_google_drive(id, destination):
    if os.path.exists(destination) and os.path.getsize(destination) > 100000:
        print(f"Already exists: {destination} ({os.path.getsize(destination)/(1024*1024):.2f} MB)")
        return
    
    URL = "https://docs.google.com/uc?export=download&confirm=t"
    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    CHUNK_SIZE = 65536
    total_size = 0
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                total_size += len(chunk)
    print(f"Downloaded {destination} ({total_size / (1024*1024):.2f} MB)")

for name, fid in videos.items():
    dest = os.path.join('data/raw', name)
    try:
        download_file_from_google_drive(fid, dest)
    except Exception as e:
        print(f"Error downloading {name}: {e}")
print("All pilot videos checked/downloaded.")
