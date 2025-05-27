import os
import subprocess

def main():
    if os.path.exists('stylegan-encoder'):
        print("stylegan-encoder already cloned.")
    else:
        subprocess.run(['git', 'clone', 'https://github.com/Nimalesh/stylegan-encoder.git'])
    os.makedirs('stylegan-encoder/raw_images', exist_ok=True)
    os.makedirs('stylegan-encoder/aligned_images', exist_ok=True)
    print("stylegan-encoder setup complete.")

if __name__ == "__main__":
    main()
