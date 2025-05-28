import os
import subprocess
import sys

def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

def main():
    # Clone StyleCLIP if not already present
    if not os.path.exists('StyleCLIP'):
        subprocess.run(['git', 'clone', 'https://github.com/Nimalesh/StyleCLIP.git'])
    os.chdir('StyleCLIP')

    # Install requirements
    pip_install("ftfy")
    pip_install("regex")
    pip_install("tqdm")
    pip_install("git+https://github.com/openai/CLIP.git")
    pip_install("gdown")

    # Download weights (with gdown)
    if not os.path.exists('stylegan2-ffhq-config-f.pt'):
        subprocess.run(['gdown', '--id', '1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT', '-O', 'stylegan2-ffhq-config-f.pt'])
    if not os.path.exists('model_ir_se50.pth'):
        subprocess.run(['gdown', '--id', '1N0MZSqPRJpLfP4mFQCS14ikrVSe8vQlL', '-O', 'model_ir_se50.pth'])
    print("StyleCLIP setup complete!")

if __name__ == "__main__":
    main()
