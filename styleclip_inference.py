import os
import subprocess
import sys

def main():
    # Clone StyleCLIP if not already present
    if not os.path.exists('StyleCLIP'):
        subprocess.run(['git', 'clone', 'https://github.com/Nimalesh/StyleCLIP.git'])
    os.chdir('StyleCLIP')

    # Install requirements
    subprocess.run([sys.executable, "-m", "pip", "install", "ftfy", "regex", "tqdm"])
    subprocess.run([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"])

    # Download weights if missing
    def download(url, out):
        if not os.path.exists(out):
            subprocess.run(['wget', url, '-O', out])

    # GDrive helper for big files
    def gdown(id, out):
        if not os.path.exists(out):
            subprocess.run(['gdown', '--id', id, '-O', out])

    try:
        import gdown
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown"])

    gdown('1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT', 'stylegan2-ffhq-config-f.pt')
    gdown('1N0MZSqPRJpLfP4mFQCS14ikrVSe8vQlL', 'model_ir_se50.pth')

    print("StyleCLIP setup complete!")

if __name__ == "__main__":
    main()
