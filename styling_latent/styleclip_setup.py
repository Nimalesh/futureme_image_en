import subprocess
import os
import sys
import requests

def download_file_from_google_drive(id, destination):
    # This is a simple downloader for public GDrive files
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
    print(f"Downloaded {destination}")

def main():
    # Clone StyleCLIP if not already present
    if not os.path.isdir("StyleCLIP"):
        print("Cloning StyleCLIP repository...")
        subprocess.run(["git", "clone", "https://github.com/Nimalesh/StyleCLIP.git"])
    else:
        print("StyleCLIP repo already present.")

    # Install requirements
    print("Installing requirements...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    # Download weights if not present
    weights = {
        "stylegan2-ffhq-config-f.pt": "1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT",
        "model_ir_se50.pth": "1N0MZSqPRJpLfP4mFQCS14ikrVSe8vQlL"
    }
    os.makedirs("StyleCLIP", exist_ok=True)
    for filename, fileid in weights.items():
        outpath = os.path.join("StyleCLIP", filename)
        if not os.path.exists(outpath):
            print(f"Downloading {filename}...")
            download_file_from_google_drive(fileid, outpath)
        else:
            print(f"{filename} already exists.")

    print("Setup complete! You are ready to run StyleCLIP.")

if __name__ == "__main__":
    main()
